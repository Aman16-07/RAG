"""Streamlit UI for the multimodal RAG system.

Supports:
  - **Car Image Retrieval** – search car images with text or a reference image.
  - **Document Upload** – upload and index PDF, DOCX, TXT, or Markdown files.
  - **Document Q&A** – ask questions answered from uploaded document context.

Launch with::

    streamlit run app.py
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from src.query import ImageRetriever
from src.document_rag.qa_pipeline import DocumentQAPipeline


st.set_page_config(page_title="Multimodal RAG System", layout="wide")
st.title("Multimodal RAG System — Images & Documents")


# ── Cached resources ──────────────────────────────────────────────────────


@st.cache_resource
def get_image_retriever(
    index_path: str,
    metadata_path: str,
    model_name: str,
    pretrained: str,
    device: str | None,
    nprobe: int | None,
) -> ImageRetriever:
    return ImageRetriever(
        index_path=Path(index_path),
        metadata_path=Path(metadata_path),
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        nprobe=nprobe,
    )


@st.cache_resource
def get_doc_pipeline(
    embedding_model: str,
    generation_model: str,
    device: str | None,
) -> DocumentQAPipeline:
    return DocumentQAPipeline(
        index_path=Path("vector_db/document_index.faiss"),
        metadata_path=Path("vector_db/document_metadata.json"),
        embedding_model=embedding_model,
        generation_model=generation_model,
        device=device,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Image Retrieval Settings")
    index_path = st.text_input("FAISS index path", "faiss_index/car_images.index")
    metadata_path = st.text_input("Metadata path", "embeddings/image_metadata.json")
    model_name = st.text_input("CLIP model name", "ViT-B-32")
    pretrained = st.text_input("Pretrained", "laion2b_s34b_b79k")
    device = st.selectbox("Device", options=["auto", "cuda", "cpu"], index=0)
    nprobe_input = st.number_input("IVF nprobe (optional)", min_value=0, value=0, step=1)
    top_k_images = st.slider("Top-K (images)", min_value=1, max_value=20, value=6)

    st.divider()

    st.header("Document RAG Settings")
    doc_embedding_model = st.text_input("Embedding model", "all-MiniLM-L6-v2")
    doc_generation_model = st.text_input("Generation model", "google/flan-t5-base")
    top_k_docs = st.slider("Top-K (documents)", min_value=1, max_value=20, value=5)

resolved_device = None if device == "auto" else device
nprobe = None if nprobe_input == 0 else int(nprobe_input)

# Load document pipeline (safe — creates empty index if none exists)
doc_pipeline = get_doc_pipeline(
    embedding_model=doc_embedding_model,
    generation_model=doc_generation_model,
    device=resolved_device,
)


# ── Tabs ──────────────────────────────────────────────────────────────────

tab_images, tab_upload, tab_qa = st.tabs(
    ["Car Image Retrieval", "Upload Documents", "Document Q&A"]
)


# ── Tab 1: Image Retrieval ────────────────────────────────────────────────

with tab_images:
    st.subheader("Search car images using text or a reference image")

    try:
        retriever = get_image_retriever(
            index_path=index_path,
            metadata_path=metadata_path,
            model_name=model_name,
            pretrained=pretrained,
            device=resolved_device,
            nprobe=nprobe,
        )
    except Exception as exc:
        st.error(f"Image retriever unavailable: {exc}")
        st.info("Run the image ingestion pipeline first (see README).")
        retriever = None

    if retriever is not None:
        sub_text, sub_image = st.tabs(["Text Query", "Image Query"])

        with sub_text:
            text_query = st.text_input("Describe the car to retrieve", "red sports car")
            if st.button("Search by Text", type="primary"):
                start = time.perf_counter()
                results = retriever.search_by_text(text_query, top_k=top_k_images)
                elapsed_ms = (time.perf_counter() - start) * 1000

                st.caption(f"Retrieved {len(results)} results in {elapsed_ms:.1f} ms")
                cols = st.columns(3)
                for idx, result in enumerate(results):
                    with cols[idx % 3]:
                        st.image(
                            result.file_path,
                            caption=f"#{result.rank} | score={result.score:.4f}",
                        )
                        st.code(result.file_path, language=None)

        with sub_image:
            uploaded = st.file_uploader(
                "Upload a car image",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                key="image_query_upload",
            )
            if uploaded and st.button("Search by Image", type="primary"):
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Query image", width=300)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                    tmp_path = Path(temp.name)
                    image.save(tmp_path)

                try:
                    start = time.perf_counter()
                    results = retriever.search_by_image(tmp_path, top_k=top_k_images)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    st.caption(f"Retrieved {len(results)} results in {elapsed_ms:.1f} ms")

                    cols = st.columns(3)
                    for idx, result in enumerate(results):
                        with cols[idx % 3]:
                            st.image(
                                result.file_path,
                                caption=f"#{result.rank} | score={result.score:.4f}",
                            )
                            st.code(result.file_path, language=None)
                finally:
                    tmp_path.unlink(missing_ok=True)


# ── Tab 2: Document Upload ────────────────────────────────────────────────

with tab_upload:
    st.subheader("Upload and index documents")
    st.caption("Supported formats: PDF, DOCX, TXT, Markdown")

    uploaded_files = st.file_uploader(
        "Choose document files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key="doc_upload",
    )

    if uploaded_files and st.button("Upload & Index", type="primary"):
        docs_dir = Path("documents")
        docs_dir.mkdir(exist_ok=True)

        total_new_chunks = 0
        for uploaded_file in uploaded_files:
            file_path = docs_dir / uploaded_file.name
            file_path.write_bytes(uploaded_file.getbuffer())
            with st.spinner(f"Processing {uploaded_file.name}..."):
                n_chunks = doc_pipeline.ingest_document(file_path)
            if n_chunks > 0:
                st.success(f"{uploaded_file.name} — {n_chunks} chunks indexed")
                total_new_chunks += n_chunks
            else:
                st.info(f"{uploaded_file.name} already indexed (skipped)")

        if total_new_chunks > 0:
            st.success(
                f"Done! {total_new_chunks} new chunks added. "
                f"Total: {doc_pipeline.total_chunks}"
            )

    # Show indexed documents summary
    if doc_pipeline.total_chunks > 0:
        st.divider()
        st.metric("Total indexed chunks", doc_pipeline.total_chunks)
        indexed = doc_pipeline.indexed_documents
        if indexed:
            st.write("**Indexed documents:**")
            for name in sorted(indexed):
                st.write(f"- {name}")


# ── Tab 3: Document Q&A ──────────────────────────────────────────────────

with tab_qa:
    st.subheader("Ask questions about uploaded documents")

    question = st.text_input("Enter your question", key="doc_question")

    if st.button("Get Answer", type="primary") and question:
        with st.spinner("Searching documents and generating answer..."):
            start = time.perf_counter()
            answer, results = doc_pipeline.ask(question, top_k=top_k_docs)
            elapsed_ms = (time.perf_counter() - start) * 1000

        st.caption(f"Answered in {elapsed_ms:.1f} ms")

        st.markdown("### Answer")
        st.markdown(answer)

        if results:
            st.divider()
            st.markdown("### Retrieved Context")
            for r in results:
                with st.expander(
                    f"#{r.rank} | {r.document_source} | score={r.score:.4f}"
                ):
                    st.markdown(r.text_content)
