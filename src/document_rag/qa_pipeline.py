"""End-to-end document question-answering pipeline.

Orchestrates document loading, chunking, embedding, indexing, retrieval,
and answer generation.  Documents are indexed once and persisted; subsequent
queries re-use the saved FAISS index.
"""
from __future__ import annotations

from pathlib import Path

from .chunker import chunk_text
from .document_loader import SUPPORTED_DOC_EXTENSIONS, load_document
from .document_retrieval import DocumentIndex, DocumentSearchResult
from .embed_documents import DocumentEmbedder


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
        query_embedding = self.embedder.embed_query(question)
        results = self.doc_index.search(query_embedding, top_k=top_k)

        if not results:
            return "No relevant documents found. Please upload documents first.", []

        context = "\n\n".join(
            f"[Source: {r.document_source}]\n{r.text_content}" for r in results
        )

        prompt = (
            f"Answer the question based on the following context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        generator = self._get_generator()
        if generator is not None:
            try:
                output = generator(prompt, max_new_tokens=512)
                answer = output[0]["generated_text"]
            except Exception:
                answer = self._format_context_answer(results)
        else:
            answer = self._format_context_answer(results)

        return answer, results

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
