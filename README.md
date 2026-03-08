# Multimodal RAG System — Images & Documents (CLIP + FAISS + SentenceTransformers)

A production-quality **Retrieval Augmented Generation** system combining **car image retrieval** and **document question answering**.

- **Image RAG** — Given a natural-language query such as *"red sports car"* or *"luxury black sedan"*, the system returns the most visually similar car images from the indexed dataset in under 100 ms.
- **Document RAG** — Upload PDF, DOCX, TXT, or Markdown files, then ask natural-language questions. The system retrieves relevant chunks and generates answers grounded in your documents.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Image RAG Pipeline](#image-rag-pipeline)
4. [Document RAG Pipeline](#document-rag-pipeline)
5. [Dataset](#dataset)
6. [Installation](#installation)
7. [Running the Project](#running-the-project)
8. [Example Queries](#example-queries)
9. [Performance Optimisation](#performance-optimisation)
10. [Future Improvements](#future-improvements)

---

## Project Overview

This project implements a **multimodal RAG system** with two independent retrieval pipelines:

| Pipeline | Capability | Technology |
|---|---|---|
| **Image RAG** | Image & text embeddings, image retrieval | [OpenCLIP](https://github.com/mlfoundations/open_clip) (`ViT-B-32`) |
| **Document RAG** | Document ingestion, chunking, QA | [SentenceTransformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| **Vector search** | Similarity search | [FAISS](https://github.com/facebookresearch/faiss) (Flat / IVF) |
| **Answer generation** | Context-grounded answers | [HuggingFace Transformers](https://huggingface.co/docs/transformers) (`flan-t5-base`) |
| **Interactive UI** | Unified interface | [Streamlit](https://streamlit.io/) |

Key design goals:

- **Low latency** – retrieval under 100 ms after model warm-up.
- **High accuracy** – L2-normalised embeddings with cosine similarity.
- **Modular architecture** – each pipeline stage is a standalone module.
- **Persistence** – documents are indexed once and remain available across sessions.
- **Maintainability** – clean separation between image and document pipelines.

---

## Architecture

```text
project/
│
├── dataset/
│   └── images/                        # Raw car images
│
├── documents/                         # Uploaded documents (persistent)
│
├── embeddings/
│   ├── manifest.json                  # Image manifest (paths + metadata)
│   ├── image_embeddings.npy           # Cached CLIP embedding matrix
│   └── image_metadata.json            # Metadata aligned with embeddings
│
├── faiss_index/
│   └── car_images.index               # Persisted FAISS index (images)
│
├── vector_db/
│   ├── document_index.faiss           # Persisted FAISS index (documents)
│   └── document_metadata.json         # Chunk metadata for documents
│
├── src/
│   ├── __init__.py
│   ├── clip_backend.py                # OpenCLIP encoder (images & text)
│   ├── embed.py                       # Batch embedding generation CLI
│   ├── faiss_store.py                 # FAISS index build / save / load
│   ├── index.py                       # Index builder CLI
│   ├── ingest.py                      # Image data ingestion & metadata
│   ├── query.py                       # Image retrieval pipeline & CLI
│   ├── utils.py                       # Shared helpers
│   │
│   └── document_rag/                  # Document RAG package
│       ├── __init__.py
│       ├── document_loader.py         # Parse PDF, DOCX, TXT, MD
│       ├── chunker.py                 # Text chunking with overlap
│       ├── embed_documents.py         # SentenceTransformer embeddings
│       ├── document_retrieval.py      # FAISS document index & search
│       └── qa_pipeline.py            # End-to-end QA orchestrator
│
├── app.py                             # Streamlit interactive UI
├── requirements.txt
└── README.md
```

### Module Responsibilities

#### Image RAG

| Module | Purpose |
|---|---|
| `ingest.py` | Discover images, validate with PIL, extract metadata, write manifest. |
| `clip_backend.py` | Wrap OpenCLIP model; `encode_images()` (batched), `encode_text()`, `encode_query_image()`. |
| `embed.py` | Read manifest → generate CLIP embeddings → save `.npy` + metadata JSON. |
| `faiss_store.py` | Build FAISS Flat or IVF index, persist / load from disk. |
| `index.py` | CLI that loads embeddings and writes the FAISS index file. |
| `query.py` | `ImageRetriever` class: load index + metadata, encode query, return ranked results. |
| `utils.py` | File discovery, JSON I/O, L2 normalisation, chunk iterator. |

#### Document RAG

| Module | Purpose |
|---|---|
| `document_loader.py` | Extract text from PDF (pdfplumber), DOCX (python-docx), TXT, and MD files. |
| `chunker.py` | Split text into overlapping word-level chunks (~500 tokens, 50-token overlap). |
| `embed_documents.py` | Encode text chunks with SentenceTransformers; L2-normalised for FAISS IP search. |
| `document_retrieval.py` | Persistent FAISS `IndexFlatIP` with incremental add, top-K search, metadata tracking. |
| `qa_pipeline.py` | End-to-end orchestrator: ingest → chunk → embed → index → retrieve → generate answer. |

---

## Image RAG Pipeline

```
Car Images Dataset
        │
        ▼
┌──────────────────┐
│  1. Ingestion    │  Scan images, validate, extract metadata
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  2. Embedding    │  CLIP encodes images in batches → .npy
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  3. Indexing     │  Build FAISS index (Flat or IVF) → .index
└────────┬─────────┘
         │
         ▼  (offline pipeline complete — artefacts cached to disk)
         │
   ┌─────┴──────┐
   │ User Query │  "red sports car"
   └─────┬──────┘
         │
         ▼
┌──────────────────┐
│  4. Query Embed  │  CLIP encodes text → query vector
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  5. Similarity   │  FAISS inner-product search → top-K indices + scores
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  6. Retrieval    │  Map indices → metadata → image paths → display
└──────────────────┘
```

---

## Document RAG Pipeline

```
Document Upload (PDF / DOCX / TXT / MD)
        │
        ▼
┌──────────────────┐
│  1. Parsing      │  Extract text with pdfplumber / python-docx / file I/O
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  2. Chunking     │  Split into ~500-token chunks with 50-token overlap
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  3. Embedding    │  SentenceTransformer encodes chunks → L2-normalised vectors
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  4. Indexing     │  Add vectors to FAISS IndexFlatIP → persist to disk
└────────┬─────────┘
         │
         ▼  (ingestion complete — index persisted)
         │
   ┌─────┴──────┐
   │ User Q     │  "What does the report say about safety?"
   └─────┬──────┘
         │
         ▼
┌──────────────────┐
│  5. Query Embed  │  SentenceTransformer encodes question
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  6. Retrieval    │  FAISS search → top-K relevant chunks
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  7. Generation   │  Assemble context + question → LLM → answer
└──────────────────┘
```

### Document Chunking Strategy

Documents are split into overlapping word-level chunks:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 500 | Maximum words per chunk (approximates tokens) |
| `overlap` | 50 | Words shared between consecutive chunks |

Overlap ensures that information at chunk boundaries is not lost during retrieval.

### Answer Generation

Retrieved chunks are assembled into a prompt and passed to a text-generation model:

1. **Primary**: HuggingFace `text2text-generation` pipeline (default: `google/flan-t5-base`)
2. **Fallback**: If the generation model is unavailable, the system returns the retrieved chunks formatted as a context-based answer

### Document Persistence

- Uploaded files are stored in `documents/`
- The FAISS index is saved at `vector_db/document_index.faiss`
- Chunk metadata is saved at `vector_db/document_metadata.json`
- Documents are indexed once — re-uploading the same file is a no-op

---

## Dataset

The image pipeline works with any directory of car images.  Recommended public datasets:

| Dataset | Source |
|---|---|
| Stanford Cars | https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset |
| CompCars | http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/ |

### Expected Structure

```
dataset/
  images/
    car_001.jpg
    car_002.jpg
    ...
```

Sub-directories are supported — folder names become **category** metadata:

```
dataset/
  images/
    sedan/
      img_001.jpg
    suv/
      img_002.jpg
```

---

## Installation

### Prerequisites

- Python 3.10+
- (Optional) NVIDIA GPU with CUDA for accelerated inference

### Steps

```bash
# 1. Clone the repository
git clone <repo-url> && cd RAG

# 2. Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU users:** install the CUDA-enabled PyTorch build *before* the requirements:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> Then install the rest: `pip install -r requirements.txt`

---

## Running the Project

### Image RAG Pipeline

The image pipeline is executed in four sequential stages.  Each stage is idempotent.

#### 1. Ingest Images

```bash
python -m src.ingest \
  --dataset-dir dataset/images \
  --manifest-path embeddings/manifest.json \
  --validate \
  --chunk-size 512
```

#### 2. Generate Embeddings

```bash
python -m src.embed \
  --manifest-path embeddings/manifest.json \
  --embeddings-path embeddings/image_embeddings.npy \
  --metadata-path embeddings/image_metadata.json \
  --batch-size 64
```

#### 3. Build FAISS Index

```bash
# Flat index (best accuracy, brute-force)
python -m src.index --index-type flat --index-path faiss_index/car_images.index

# IVF index (faster on large datasets)
python -m src.index --index-type ivf --nlist 256 --index-path faiss_index/car_images.index
```

#### 4. Query (CLI)

```bash
python -m src.query --text-query "red sports car" --top-k 5
python -m src.query --image-query dataset/images/example.jpg --top-k 5
```

### Document RAG Pipeline

The document pipeline is managed entirely through the Streamlit UI:

1. Navigate to the **Upload Documents** tab
2. Upload one or more PDF, DOCX, TXT, or MD files
3. Click **Upload & Index** — documents are parsed, chunked, embedded, and indexed
4. Switch to the **Document Q&A** tab
5. Type a question and click **Get Answer**

Indexed documents persist across sessions in `vector_db/`.

### Streamlit UI

```bash
streamlit run app.py
```

The UI provides three tabs:

| Tab | Function |
|---|---|
| **Car Image Retrieval** | Text and image query search for car images |
| **Upload Documents** | Upload and index PDF / DOCX / TXT / MD files |
| **Document Q&A** | Ask questions answered from document context |

---

## Example Queries

### Image Retrieval

| Query | Expected Results |
|---|---|
| `red sedan` | Images of red sedan-style cars |
| `white SUV` | Images of white sport-utility vehicles |
| `blue sports car` | Images of blue sports / performance cars |
| `luxury black sedan` | Images of black luxury sedans |

### Document Q&A

| Query | Example |
|---|---|
| `What are the key findings?` | Retrieves and summarises findings from uploaded reports |
| `Summarise the safety section` | Finds safety-related chunks and generates a summary |
| `What does the contract say about liability?` | Retrieves relevant legal clauses |

---

## Performance Optimisation

| Technique | Where | Impact |
|---|---|---|
| **Batch embedding** | `clip_backend.py` / `embed.py` / `embed_documents.py` | Maximises GPU utilisation during index build. |
| **Persistent FAISS index** | `faiss_store.py` / `document_retrieval.py` | Index is written once and loaded in milliseconds at query time. |
| **Cached embeddings** | `embeddings/image_embeddings.npy` | Avoids re-encoding images on every run. |
| **Idempotent ingestion** | `qa_pipeline.py` | Documents are indexed once; duplicates are skipped. |
| **L2 normalisation** | `utils.py` / `clip_backend.py` / `embed_documents.py` | Enables cosine similarity via fast inner-product. |
| **GPU auto-detection** | `clip_backend.py` / SentenceTransformers | Uses CUDA when available; falls back to CPU. |
| **IVF index** | `faiss_store.py` | Sub-linear search on large image collections. |
| **`@torch.inference_mode()`** | `clip_backend.py` | Disables autograd overhead during encoding. |
| **Streamlit `@cache_resource`** | `app.py` | Models + indices loaded once per server session. |
| **Lazy LLM loading** | `qa_pipeline.py` | Generation model loaded only when first question is asked. |
| **Incremental index** | `document_retrieval.py` | New documents appended to existing FAISS index without rebuild. |

### Latency Targets

| Stage | Typical Latency |
|---|---|
| Text query encoding (CLIP) | ~5–15 ms (GPU) / ~30–80 ms (CPU) |
| Text query encoding (SentenceTransformer) | ~5–10 ms (GPU) / ~20–50 ms (CPU) |
| FAISS search (Flat, 10k vectors) | < 1 ms |
| Document chunk retrieval | < 5 ms |
| Answer generation (flan-t5-base) | ~200–500 ms (GPU) / ~1–3 s (CPU) |
| **End-to-end image retrieval** | **< 100 ms** (GPU) |

---

## Future Improvements

- **Metadata-aware re-ranking** – combine CLIP similarity with structured metadata (make, model, year) for higher precision.
- **HNSW index** – experiment with `IndexHNSWFlat` for better recall/latency on million-scale datasets.
- **Multi-GPU / ONNX export** – export encoders to ONNX for deterministic, hardware-agnostic inference.
- **Hybrid search** – fuse keyword search (BM25) with vector similarity for document retrieval.
- **Streaming generation** – stream LLM answers token-by-token in the Streamlit UI.
- **Configurable LLM backend** – support OpenAI API, Ollama, or other LLM providers for answer generation.
- **Cross-modal search** – query images using document context or vice versa.
- **REST API** – wrap both retrievers behind FastAPI endpoints.
- **Evaluation harness** – measure Recall@K, MRR, and nDCG against labelled test splits.
- **Embedding quantisation** – use `faiss.IndexScalarQuantizer` or PQ to reduce memory footprint.
- **Docker container** – provide a `Dockerfile` for one-command deployment.
- **Table and image extraction from PDFs** – extract structured tables and embedded images from documents.


