# tutor_rag_pipeline (RAG-Anything)

A multimodal Retrieval-Augmented Generation (RAG) pipeline for educational content. It ingests PDFs and images, classifies and enhances content (text / tables / equations / images) using vision and language models, embeds it, indexes it in Milvus, and generates study materials such as Q&A pairs, quizzes, descriptions, and questionnaires.

## Features

- **Multimodal ingestion** — parses PDFs (PyMuPDF with optional OCR/Tesseract, or RAGAnything backend) and image files
- **Content classification** — heuristic detection of text, tables, equations, and images
- **LLM enhancement** — OpenRouter-backed vision and language models describe, analyze, and summarize content
- **Vector search** — Ollama `nomic-embed-text` embeddings stored in Milvus (HNSW/COSINE) with content-type, source, and chapter filters
- **Study material generation** — Q&A pairs, quizzes, descriptions, and questionnaires from indexed content
- **REST API** — FastAPI server exposing processing, search, and generation endpoints
- **Performance monitoring** — timing and success-rate tracking for pipeline stages

## Pipeline

```
PDF / Image → Parse → Classify (text|table|equation|image) → Enhance (LLM/Vision)
    → Embed (Ollama nomic-embed-text) → Store (Milvus) → Retrieve → Generate study materials
```

## Directory Structure

```
main.py                      # CLI entry point
src/
  api/server.py              # FastAPI REST API
  config/settings.py         # Environment-based configuration
  processors/                # Modality processors (image, table, equation, generic)
  rag/                       # Parser, orchestrator, embeddings, LLM client,
                             # Milvus storage, QA/quiz/questionnaire generators,
                             # performance monitor
  utils/                     # File handling and exceptions
tests/                       # Unit tests
```

## Prerequisites

- Python >= 3.12 and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running with the `nomic-embed-text` model pulled (`ollama pull nomic-embed-text`)
- [Milvus](https://milvus.io/) (standalone or Zilliz Cloud) reachable at `MILVUS_URI`
- An [OpenRouter](https://openrouter.ai/) API key for LLM/vision calls

## Setup

```bash
uv sync
export OPENROUTER_API_KEY=your_key   # or set via a local .env file
ollama pull nomic-embed-text
```

## Environment Variables

All settings are read from the environment or a `.env` file. Key variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — (required) | OpenRouter API key for LLM/vision calls |
| `MILVUS_URI` / `MILVUS_TOKEN` | `localhost:19530` | Milvus connection |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_NOMIC_MODEL` | `nomic-embed-text` | Embedding model |
| `PARSER` | `pymupdf` | Parser backend (`pymupdf` or `raganything`) |
| `ENABLE_OCR` | `false` | Enable Tesseract OCR for scanned PDFs |
| `ENABLE_IMAGE/TABLE/EQUATION_PROCESSING` | `true` | Toggle multimodal processing |
| `SEMANTIC_CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Chunking parameters |
| `WORKING_DIR` | `./rag_storage` | RAG working directory |

## Usage

### CLI

Place PDFs/images in a `test_pdfs/` directory (gitignored), then:

```bash
python main.py
```

This processes the directory, uploads embeddings to Milvus, and prints generated questionnaires.

### API

```bash
uvicorn src.api.server:app --port 8001
```

Main endpoints: `/health`, `/process/{document,documents,directory}`, `/search*`, `/qa/generate`, `/quiz/generate`, `/description/generate`, `/learning-material/generate`, `/questionnaires/generate`, `/performance/*`.

## Tests

```bash
uv run pytest
```

## Security

No secrets are stored in this repository. All credentials are supplied via environment variables (`.env` is gitignored).