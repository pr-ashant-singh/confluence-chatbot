# confluence-chatbot

Turn your Confluence docs into a searchable AI chatbot in 5 minutes.

`confluence-chatbot` ingests your Confluence spaces — text, tables, and architecture diagrams — into a local vector store, then answers questions using RAG (Retrieval-Augmented Generation) with source attribution.

## Features

- **Smart chunking** — splits by HTML headings, keeps tables whole, handles unstructured pages with paragraph-based fallback
- **Page context in embeddings** — each chunk is prefixed with page title + section heading for better retrieval
- **Diagram understanding** — uses a vision model (LLaVA or Claude Vision) to describe architecture diagrams as searchable text
- **Fully local** — runs with FAISS + Ollama, zero cloud costs for development
- **Production-ready** — swap to S3 Vectors + Bedrock for AWS deployment
- **Incremental sync** — only re-processes pages that changed since last sync
- **Source attribution** — every answer cites exactly which page and section it came from
- **Pluggable architecture** — swap embedding models, vector stores, and LLMs without changing code

## Quick Start

### Install

```bash
pip install confluence-chatbot[all]
```

### Configure

Create a `.env` file (see `.env.example`):

```env
CONFLUENCE_URL=https://yourcompany.atlassian.net
CONFLUENCE_EMAIL=you@company.com
CONFLUENCE_API_TOKEN=your-api-token
```

### Prerequisites

```bash
# Install and start Ollama (for local LLM)
brew install ollama
ollama serve  # in a separate terminal

# Pull the models
ollama pull llama3.1:8b    # for answer generation
ollama pull llava:13b      # for diagram description (optional)
```

### Use as a Library

```python
from confluence_chatbot import ConfluenceChatbot

rag = ConfluenceChatbot(
    confluence_url="https://yourcompany.atlassian.net",
    confluence_email="you@company.com",
    confluence_token="your-token",
    vector_store="faiss",
    embedding_model="BAAI/bge-large-en-v1.5",
    llm="ollama/llama3.1:8b",
)

# Ingest a space
rag.sync(spaces=["ENG"])

# Ask questions
answer = rag.ask("How does our caching layer work?")
print(answer.text)
print(answer.sources)
```

### Use as CLI

```bash
# Sync a Confluence space
confluence-chatbot sync --space ENG

# Sync specific pages
confluence-chatbot sync --page-id 1234567890

# Force full re-sync (ignore cached state)
confluence-chatbot sync --space ENG --full

# Ask a question
confluence-chatbot ask "How does caching work?"

# Interactive mode
confluence-chatbot ask
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vector_store` | `"faiss"` | `"faiss"` (local) or `"s3vectors"` (AWS) |
| `embedding_model` | `"BAAI/bge-large-en-v1.5"` | Any sentence-transformers model or `"bedrock/titan"` |
| `llm` | `"ollama/llama3.1:8b"` | Ollama model or `"bedrock/claude"` |
| `enable_image_description` | `False` | Describe diagrams with vision model |
| `image_model` | `"llava:13b"` | Vision model for diagrams |
| `top_k` | `8` | Number of chunks to retrieve per query |
| `s3_bucket_name` | `""` | S3 Vectors bucket (required for s3vectors) |
| `s3_region` | `"us-east-1"` | AWS region for S3 Vectors |
| `s3_profile` | `None` | AWS CLI profile name |

## How It Works

```
Confluence → Fetch pages → Smart chunk (text/tables/images)
                                    ↓
                         Embed (BAAI/bge-large-en-v1.5 or Bedrock Titan)
                                    ↓
                         Store in FAISS (local) or S3 Vectors (AWS)
                                    ↓
User question → Embed → Similarity search → Top-K chunks
                                    ↓
                         LLM generates grounded answer
                                    ↓
                         Answer + source citations
```

## Install Options

```bash
# Lightweight (no local models, for use with Bedrock APIs)
pip install confluence-chatbot

# With local embedding model (BAAI, requires PyTorch ~800MB)
pip install confluence-chatbot[local]

# With local LLM via Ollama
pip install confluence-chatbot[ollama]

# Full local setup (recommended for development)
pip install confluence-chatbot[all]
```

## Project Structure

```
src/confluence_chatbot/
├── core.py                  # Main ConfluenceChatbot orchestrator
├── models.py                # Data models (Page, Chunk, Answer)
├── cli.py                   # Command-line interface
├── ingest/
│   ├── confluence_client.py # Confluence API integration
│   ├── html_parser.py       # Parse Confluence HTML
│   ├── chunker.py           # Smart content-aware chunking
│   ├── image_describer.py   # Vision model for diagrams
│   └── sync_manager.py      # Incremental sync state tracking
├── embedding/
│   ├── base.py              # Abstract embedding interface
│   ├── sentence_transformer.py  # Local embedding (BAAI)
│   └── bedrock.py           # AWS Bedrock Titan Embeddings
├── vector_store/
│   ├── base.py              # Abstract vector store interface
│   ├── faiss_store.py       # Local FAISS implementation
│   └── s3_vectors.py        # AWS S3 Vectors implementation
└── generation/
    ├── base.py              # Abstract LLM interface
    ├── ollama_llm.py        # Local Ollama implementation
    └── bedrock_llm.py       # AWS Bedrock Claude implementation
```

## Roadmap

- [x] v0.1 — Core pipeline: sync, chunk, embed, query, answer (local)
- [x] v0.2 — Incremental sync, Bedrock LLM + embeddings, improved chunking
- [ ] v0.3 — Slack/Google Chat integration examples, evaluation tooling
- [ ] v1.0 — Production deployment guide, CDK infrastructure template

## License

MIT
