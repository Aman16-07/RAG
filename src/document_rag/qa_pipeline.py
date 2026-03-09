"""End-to-end document question-answering pipeline.

Orchestrates document loading, chunking, embedding, indexing, retrieval,
and answer generation.  Documents are indexed once and persisted; subsequent
queries re-use the saved FAISS index.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.telemetry import get_meter, get_tracer

from .chunker import chunk_text
from .document_loader import SUPPORTED_DOC_EXTENSIONS, load_document
from .document_retrieval import DocumentIndex, DocumentSearchResult
from .embed_documents import DocumentEmbedder

_tracer = get_tracer("document-rag")
_meter = get_meter("document-rag")

# OTel metrics instruments
_query_latency_hist = _meter.create_histogram(
    "rag.doc.query_latency_ms",
    description="Total document query latency in ms",
    unit="ms",
)
_embedding_latency_hist = _meter.create_histogram(
    "rag.doc.embedding_latency_ms",
    description="Query embedding latency in ms",
    unit="ms",
)
_vector_search_latency_hist = _meter.create_histogram(
    "rag.doc.vector_search_latency_ms",
    description="Vector search latency in ms",
    unit="ms",
)
_answer_gen_latency_hist = _meter.create_histogram(
    "rag.doc.answer_generation_latency_ms",
    description="Answer generation latency in ms",
    unit="ms",
)
_retrieval_accuracy_gauge = _meter.create_histogram(
    "rag.doc.retrieval_accuracy_percentage",
    description="Retrieval accuracy percentage (mean similarity * 100)",
    unit="%",
)
_vector_index_size_gauge = _meter.create_histogram(
    "rag.doc.vector_index_size",
    description="Number of vectors in the FAISS index at query time",
)


@dataclass
class QueryReport:
    """Evaluation metrics produced by a single document Q&A query."""

    answer: str
    results: list[DocumentSearchResult]

    # vector search metrics
    vectors_indexed: int = 0
    top_k: int = 0
    similarity_scores: list[float] = field(default_factory=list)
    accuracy_pct: float = 0.0

    # latency (milliseconds)
    embedding_ms: float = 0.0
    vector_search_ms: float = 0.0
    answer_generation_ms: float = 0.0
    total_ms: float = 0.0


class DocumentQAPipeline:
    """High-level API for document ingestion and question answering."""

    def __init__(
        self,
        index_path: Path = Path("vector_db/document_index.faiss"),
        metadata_path: Path = Path("vector_db/document_metadata.json"),
        embedding_model: str = "all-MiniLM-L6-v2",
        generation_model: str = "google/flan-t5-base",
        device: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.embedder = DocumentEmbedder(model_name=embedding_model, device=device)
        self.doc_index = DocumentIndex(index_path, metadata_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._generator = None
        self._generation_model = generation_model

    # ── lazy model loading ─────────────────────────────────────────────

    def _get_generator(self):
        """Lazy-load the text generation pipeline on first use."""
        if self._generator is None:
            try:
                from transformers import pipeline

                self._generator = pipeline(
                    "text2text-generation",
                    model=self._generation_model,
                    max_new_tokens=512,
                )
            except Exception:
                self._generator = None
        return self._generator

    # ── ingestion ──────────────────────────────────────────────────────

    def ingest_document(self, file_path: Path, source_name: str | None = None) -> int:
        """Ingest a document: extract text, chunk, embed, and index.

        Returns the number of chunks added.  Returns ``0`` if the document
        was already indexed (idempotent).
        """
        source = source_name or file_path.name
        if source in self.doc_index.get_indexed_documents():
            return 0

        text = load_document(file_path)
        chunks = chunk_text(
            text, source, chunk_size=self.chunk_size, overlap=self.chunk_overlap
        )
        if not chunks:
            return 0

        texts = [c.text_content for c in chunks]
        embeddings = self.embedder.embed_chunks(texts, show_progress=False)

        metadata = [
            {
                "chunk_id": c.chunk_id,
                "document_source": c.document_source,
                "text_content": c.text_content,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        self.doc_index.add(embeddings, metadata)
        return len(chunks)

    def ingest_directory(self, directory: Path) -> dict[str, int]:
        """Ingest all supported documents from a directory.

        Creates the directory if it does not exist.  Returns a dict
        mapping each filename to the number of chunks indexed (0 if
        already indexed or empty).
        """
        directory.mkdir(parents=True, exist_ok=True)
        results: dict[str, int] = {}
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_DOC_EXTENSIONS:
                results[path.name] = self.ingest_document(path)
        return results

    # ── question answering ─────────────────────────────────────────────

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> tuple[str, list[DocumentSearchResult]]:
        """Answer a question using retrieved document context.

        Returns a tuple of ``(answer_text, search_results)``.
        """
        report = self.ask_with_metrics(question, top_k=top_k)
        return report.answer, report.results

    def ask_with_metrics(
        self,
        question: str,
        top_k: int = 5,
    ) -> QueryReport:
        """Answer a question and return a full :class:`QueryReport`."""
        with _tracer.start_as_current_span("query_received", attributes={
            "query.text": question,
            "query.top_k": top_k,
        }) as root_span:
            t_total = time.perf_counter()

            # 1. Query embedding
            with _tracer.start_as_current_span("query_embedding_generation"):
                t0 = time.perf_counter()
                query_embedding = self.embedder.embed_query(question)
                embedding_ms = (time.perf_counter() - t0) * 1000

            # 2. Vector search
            with _tracer.start_as_current_span("vector_search") as vs_span:
                t0 = time.perf_counter()
                results = self.doc_index.search(query_embedding, top_k=top_k)
                vector_search_ms = (time.perf_counter() - t0) * 1000

                vectors_indexed = self.doc_index.total_chunks
                similarity_scores = [r.score for r in results]
                accuracy_pct = (
                    float(np.mean(similarity_scores) * 100)
                    if similarity_scores
                    else 0.0
                )

                vs_span.set_attribute("vector_index_size", vectors_indexed)
                vs_span.set_attribute("top_k", top_k)
                vs_span.set_attribute("results_count", len(results))
                vs_span.set_attribute("similarity_scores", similarity_scores)
                vs_span.set_attribute("vector_search_latency_ms", vector_search_ms)

            if not results:
                total_ms = (time.perf_counter() - t_total) * 1000
                root_span.set_attribute("total_latency_ms", total_ms)
                return QueryReport(
                    answer="No relevant documents found. Please upload documents first.",
                    results=[],
                    vectors_indexed=vectors_indexed,
                    top_k=top_k,
                    similarity_scores=[],
                    accuracy_pct=0.0,
                    embedding_ms=embedding_ms,
                    vector_search_ms=vector_search_ms,
                    answer_generation_ms=0.0,
                    total_ms=total_ms,
                )

            # 3. Context retrieval
            with _tracer.start_as_current_span("context_retrieval") as ctx_span:
                context = "\n\n".join(
                    f"[Source: {r.document_source}]\n{r.text_content}"
                    for r in results
                )
                ctx_span.set_attribute("context_chunks_count", len(results))
                ctx_span.set_attribute("context_tokens", len(context.split()))

            prompt = (
                f"Answer the question based on the following context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )

            # 4. Answer generation
            with _tracer.start_as_current_span("answer_generation") as gen_span:
                t0 = time.perf_counter()
                generator = self._get_generator()
                if generator is not None:
                    try:
                        output = generator(prompt, max_new_tokens=512)
                        answer = output[0]["generated_text"]
                    except Exception:
                        answer = self._format_context_answer(results)
                else:
                    answer = self._format_context_answer(results)
                answer_generation_ms = (time.perf_counter() - t0) * 1000
                gen_span.set_attribute("generation_time_ms", answer_generation_ms)
                gen_span.set_attribute("context_chunks_count", len(results))
                gen_span.set_attribute("context_tokens", len(context.split()))

            total_ms = (time.perf_counter() - t_total) * 1000

            # Record OTel metrics
            root_span.set_attribute("total_latency_ms", total_ms)
            root_span.set_attribute("retrieval_accuracy_percentage", accuracy_pct)
            _embedding_latency_hist.record(embedding_ms)
            _vector_search_latency_hist.record(vector_search_ms)
            _answer_gen_latency_hist.record(answer_generation_ms)
            _query_latency_hist.record(total_ms)
            _retrieval_accuracy_gauge.record(accuracy_pct)
            _vector_index_size_gauge.record(vectors_indexed)

            return QueryReport(
                answer=answer,
                results=results,
                vectors_indexed=vectors_indexed,
                top_k=top_k,
                similarity_scores=similarity_scores,
                accuracy_pct=accuracy_pct,
                embedding_ms=embedding_ms,
                vector_search_ms=vector_search_ms,
                answer_generation_ms=answer_generation_ms,
                total_ms=total_ms,
            )

    @staticmethod
    def _format_context_answer(results: list[DocumentSearchResult]) -> str:
        """Fallback answer formatting when no generation model is available."""
        parts = ["Based on the retrieved documents:\n"]
        for r in results:
            parts.append(f"**[{r.document_source}]** (relevance: {r.score:.4f})")
            parts.append(r.text_content)
            parts.append("")
        return "\n".join(parts)

    # ── introspection ──────────────────────────────────────────────────

    @property
    def indexed_documents(self) -> set[str]:
        """Set of document names that have been indexed."""
        return self.doc_index.get_indexed_documents()

    @property
    def total_chunks(self) -> int:
        """Total number of indexed chunks across all documents."""
        return self.doc_index.total_chunks
