# confluence-rag: Detailed Usage Guide

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Option A: Fully Local Setup (Free)](#option-a-fully-local-setup-free)
- [Option B: AWS Production Setup](#option-b-aws-production-setup)
- [Using the CLI](#using-the-cli)
- [Using as a Python Library](#using-as-a-python-library)
- [Incremental Sync](#incremental-sync)
- [Diagram Support](#diagram-support)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required (All setups)

- Python 3.11+
- A Confluence Cloud or Server instance
- Confluence API token ([Create one here](https://id.atlassian.com/manage-profile/security/api-tokens))

### For Local Setup (Option A)

- [Ollama](https://ollama.com) installed
- ~12 GB disk space (for models)

### For AWS Production Setup (Option B)

- AWS account with Bedrock access
- AWS CLI configured with SSO or access keys
- S3 Vectors bucket created

---

## Installation

### Minimal (API-only, for Bedrock backend)

```bash
pip install confluence-rag
```

### Local embeddings (BAAI model, requires PyTorch ~800MB)

```bash
pip install confluence-rag[local]
```

### Local LLM + Vision (Ollama)

```bash
pip install confluence-rag[ollama]
```

### Full local setup (recommended for development)

```bash
pip install confluence-rag[all]
```

### Development (includes linting + testing)

```bash
pip install confluence-rag[dev]
```

---

## Configuration

### Create a `.env` file

```bash
# Required — Confluence credentials
CONFLUENCE_URL=https://yourcompany.atlassian.net
CONFLUENCE_EMAIL=you@company.com
CONFLUENCE_API_TOKEN=your-api-token-here

# Optional — Override defaults for CLI
VECTOR_STORE=faiss
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
LLM_MODEL=ollama/llama3.1:8b
ENABLE_IMAGE_DESCRIPTION=false
IMAGE_MODEL=llava:13b
```

### Get a Confluence API Token

1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
2. Click "Create API token"
3. Label it (e.g., `confluence-rag`)
4. Copy the token — you won't see it again
5. Add it to your `.env` file

---

## Option A: Fully Local Setup (Free)

No cloud accounts needed. Everything runs on your machine.

### Step 1: Install

```bash
pip install confluence-rag[all]
```

### Step 2: Install and start Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama server (leave running in a separate terminal):

```bash
ollama serve
```

### Step 3: Pull the models

```bash
# LLM for answer generation (~4.7 GB, one-time download)
ollama pull llama3.1:8b

# Vision model for diagram description (~8 GB, optional)
ollama pull llava:13b
```

### Step 4: Create `.env` file

```bash
CONFLUENCE_URL=https://yourcompany.atlassian.net
CONFLUENCE_EMAIL=you@company.com
CONFLUENCE_API_TOKEN=your-token
```

### Step 5: Sync and ask

```python
from confluence_rag import ConfluenceRAG
from dotenv import load_dotenv
import os

load_dotenv()

rag = ConfluenceRAG(
    confluence_url=os.getenv("CONFLUENCE_URL"),
    confluence_email=os.getenv("CONFLUENCE_EMAIL"),
    confluence_token=os.getenv("CONFLUENCE_API_TOKEN"),
    vector_store="faiss",                      # local file storage
    embedding_model="BAAI/bge-large-en-v1.5",  # local embedding model
    llm="ollama/llama3.1:8b",                  # local LLM
)

# Ingest a Confluence space
stats = rag.sync(spaces=["ENG"])
print(f"Synced: {stats['pages_processed']} pages, {stats['chunks_created']} chunks")

# Ask questions
answer = rag.ask("How does our caching layer work?")
print(answer.text)
print(answer.sources)
```

### Where data is stored locally

```
.data/
├── faiss_index/          # Vector index (embeddings)
│   ├── index.faiss
│   ├── keys.json
│   └── metadata.json
└── sync_state.json       # Tracks which pages have been synced
```

---

## Option B: AWS Production Setup

Uses Bedrock (Titan embeddings + Claude) and S3 Vectors. Fast, scalable, managed.

### Step 1: Install

```bash
pip install confluence-rag
```

### Step 2: AWS Prerequisites

1. **AWS CLI configured:**

```bash
aws configure sso --profile your-profile
# or
aws configure --profile your-profile
```

2. **S3 Vector bucket created:**

```bash
aws s3vectors create-vector-bucket \
    --vector-bucket-name your-bucket-name \
    --profile your-profile \
    --region us-east-1
```

3. **Bedrock model access:**
   - Go to AWS Bedrock console
   - Open Model catalog
   - Select Claude Haiku 4.5 → submit use case details (one-time)
   - Titan Text Embeddings V2 is auto-enabled

### Step 3: Create `.env` file

```bash
CONFLUENCE_URL=https://yourcompany.atlassian.net
CONFLUENCE_EMAIL=you@company.com
CONFLUENCE_API_TOKEN=your-token
```

### Step 4: Sync and ask

```python
from confluence_rag import ConfluenceRAG
from dotenv import load_dotenv
import os

load_dotenv()

rag = ConfluenceRAG(
    confluence_url=os.getenv("CONFLUENCE_URL"),
    confluence_email=os.getenv("CONFLUENCE_EMAIL"),
    confluence_token=os.getenv("CONFLUENCE_API_TOKEN"),
    vector_store="s3vectors",              # AWS S3 Vectors
    embedding_model="bedrock/titan",       # Bedrock Titan Embeddings V2
    llm="bedrock/claude",                  # Claude Haiku 4.5 via Bedrock
    s3_bucket_name="your-bucket-name",
    s3_index_name="confluence-docs",
    s3_region="us-east-1",
    s3_profile="your-profile",
)

# Ingest
stats = rag.sync(spaces=["ENG"])
print(f"Synced: {stats['pages_processed']} pages, {stats['chunks_created']} chunks")

# Ask
answer = rag.ask("What AWS services does our platform use?")
print(answer.text)
```

---

## Using the CLI

The CLI reads configuration from `.env` or environment variables.

### Sync a Confluence space

```bash
# Incremental sync (only new/changed pages)
confluence-rag sync --space ENG

# Sync multiple spaces
confluence-rag sync --space ENG --space DEVOPS --space PLATFORM

# Sync specific pages by ID
confluence-rag sync --page-id 1234567890 --page-id 9876543210

# Force full re-sync (ignore cached state)
confluence-rag sync --space ENG --full
```

### Ask questions

```bash
# Direct question
confluence-rag ask "How does caching work?"

# Interactive mode (prompts for input)
confluence-rag ask

# Examples
confluence-rag ask "What is the deployment process?"
confluence-rag ask "Which databases do we use?"
confluence-rag ask "How does the auth service work?"
```

### CLI Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFLUENCE_URL` | (required) | Your Confluence instance URL |
| `CONFLUENCE_EMAIL` | (required) | Email for authentication |
| `CONFLUENCE_API_TOKEN` | (required) | API token |
| `VECTOR_STORE` | `faiss` | `faiss` or `s3vectors` |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model |
| `LLM_MODEL` | `ollama/llama3.1:8b` | LLM for answers |
| `ENABLE_IMAGE_DESCRIPTION` | `false` | Enable diagram description |
| `IMAGE_MODEL` | `llava:13b` | Vision model for diagrams |

---

## Using as a Python Library

### Basic usage

```python
from confluence_rag import ConfluenceRAG

rag = ConfluenceRAG(
    confluence_url="https://company.atlassian.net",
    confluence_email="user@company.com",
    confluence_token="token",
    vector_store="faiss",
    llm="ollama/llama3.1:8b",
)

rag.sync(spaces=["ENG"])
answer = rag.ask("How does X work?")
```

### Access the Answer object

```python
answer = rag.ask("How does caching work?")

# The generated text
print(answer.text)

# The original question
print(answer.question)

# Source citations (list of dicts)
for source in answer.sources:
    print(f"- [{source['content_type']}] {source['page_title']} > {source['heading']}")
    print(f"  URL: {source.get('page_url', '')}")
```

### Sync specific pages

```python
# By space
rag.sync(spaces=["ENG", "DEVOPS"])

# By page ID
rag.sync(spaces=[], page_ids=["1234567890", "9876543210"])

# Force full re-sync
rag.sync(spaces=["ENG"], full=True)
```

### Check sync stats

```python
stats = rag.sync(spaces=["ENG"])
print(f"Pages processed: {stats['pages_processed']}")
print(f"Pages skipped (unchanged): {stats['pages_skipped']}")
print(f"Pages deleted: {stats['pages_deleted']}")
print(f"Chunks created: {stats['chunks_created']}")
print(f"Time: {stats['time_seconds']}s")
```

### Use custom components

```python
from confluence_rag import ConfluenceRAG
from confluence_rag.embedding import SentenceTransformerEmbedding
from confluence_rag.vector_store import FAISSStore
from confluence_rag.generation import OllamaLLM

# Create components separately for more control
embedding = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
store = FAISSStore(index_path="./my-index", dimension=384)
llm = OllamaLLM(model="mistral:7b", temperature=0.2)

rag = ConfluenceRAG(
    confluence_url="...",
    confluence_email="...",
    confluence_token="...",
    vector_store=store,
    embedding_model=embedding,
    llm=llm,
)
```

---

## Incremental Sync

After the first full sync, subsequent syncs only process new or changed pages.

### How it works

1. First run: processes all pages, stores their version numbers in `.data/sync_state.json`
2. Next runs: fetches page metadata (lightweight), compares versions
3. Only fetches full content and re-embeds pages with newer versions
4. Removes vectors for deleted pages

### Example

```bash
# First run — processes everything
confluence-rag sync --space ENG
# → 50 pages processed, 127 chunks, 8.5s

# Second run — nothing changed
confluence-rag sync --space ENG
# → 0 pages processed, 50 skipped, 0.4s

# After someone edits a page
confluence-rag sync --space ENG
# → 1 page processed, 49 skipped, 1.2s
```

### Force full re-sync

```bash
confluence-rag sync --space ENG --full
```

Or in Python:

```python
rag.sync(spaces=["ENG"], full=True)
```

### State file location

The sync state is stored at `.data/sync_state.json`. Delete this file to force a full re-sync.

---

## Diagram Support

Architecture diagrams in Confluence pages can be described using a vision model and made searchable.

### Enable diagram description

```python
rag = ConfluenceRAG(
    ...,
    enable_image_description=True,
    image_model="llava:13b",  # requires Ollama running
)
```

### How it works

1. During ingestion, images are detected in the Confluence HTML (`<ac:image>` tags)
2. Images are downloaded from Confluence as attachments
3. Each image is sent to LLaVA (via Ollama) with a prompt to describe the architecture
4. The text description becomes a chunk (type: `diagram`) and gets embedded
5. When a user asks about architecture, the diagram description is retrieved and used

### Requirements

- Ollama running (`ollama serve`)
- LLaVA model pulled (`ollama pull llava:13b`)
- `enable_image_description=True` in config

### Performance note

Each diagram takes ~5-10 seconds to describe. For spaces with many diagrams, the first sync will be slower. Subsequent syncs skip unchanged pages.

---

## Configuration Reference

### ConfluenceRAG Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confluence_url` | str | (required) | Confluence instance URL |
| `confluence_email` | str | (required) | Authentication email |
| `confluence_token` | str | (required) | API token |
| `vector_store` | str or VectorStore | `"s3vectors"` | `"faiss"`, `"s3vectors"`, or custom instance |
| `embedding_model` | str or EmbeddingModel | `"BAAI/bge-large-en-v1.5"` | Model name or custom instance |
| `llm` | str or LLM | `"ollama/llama3.1:8b"` | `"ollama/<model>"`, `"bedrock/claude"`, or custom |
| `s3_bucket_name` | str | `""` | S3 Vectors bucket (required for s3vectors) |
| `s3_index_name` | str | `"confluence-docs"` | Vector index name |
| `s3_region` | str | `"us-east-2"` | AWS region |
| `s3_profile` | str or None | `None` | AWS CLI profile name |
| `enable_image_description` | bool | `False` | Describe diagrams with vision model |
| `image_model` | str | `"llava:13b"` | Ollama vision model |
| `top_k` | int | `5` | Chunks to retrieve per query |

### Supported Embedding Models

| Model String | Backend | Dimension | Notes |
|--------------|---------|-----------|-------|
| `"BAAI/bge-large-en-v1.5"` | sentence-transformers | 1024 | Best quality, local |
| `"all-MiniLM-L6-v2"` | sentence-transformers | 384 | Lighter, faster |
| `"bedrock/titan"` | Bedrock API | 1024 | Managed, pay-per-use |

### Supported LLMs

| Model String | Backend | Notes |
|--------------|---------|-------|
| `"ollama/llama3.1:8b"` | Ollama (local) | Free, decent quality |
| `"ollama/llama3.1:70b"` | Ollama (local) | Better quality, needs GPU |
| `"ollama/mistral:7b"` | Ollama (local) | Fast, good for simple Q&A |
| `"bedrock/claude"` | Bedrock API | Best quality, ~$3-4/month |

### Supported Vector Stores

| Store String | Backend | Notes |
|--------------|---------|-------|
| `"faiss"` | Local FAISS file | Free, good for <100K vectors |
| `"s3vectors"` | AWS S3 Vectors | Scalable, managed, pay-per-use |

---

## Troubleshooting

### "sentence-transformers not found"

```bash
pip install confluence-rag[local]
```

### "ollama package not found"

```bash
pip install confluence-rag[ollama]
```

### "Could not connect to Ollama"

Start the Ollama server in a separate terminal:

```bash
ollama serve
```

### "Model not found" (Ollama)

Pull the model first:

```bash
ollama pull llama3.1:8b
ollama pull llava:13b
```

### AWS SSO token expired

```bash
aws sso login --profile your-profile
```

### "Anthropic use case details required"

Go to Bedrock console → Model catalog → Claude → submit the use case form. Wait 15 minutes.

### "S3 Vector bucket not found"

Check region matches:

```bash
aws s3vectors list-vector-buckets --profile your-profile --region us-east-1
```

### Sync state is corrupted

Delete and re-sync:

```bash
rm .data/sync_state.json
confluence-rag sync --space ENG --full
```

### FAISS index is corrupted

Delete and re-sync:

```bash
rm -rf .data/faiss_index/
confluence-rag sync --space ENG --full
```

### HuggingFace rate limit warning

This is harmless. To suppress, set a HuggingFace token:

```bash
export HF_TOKEN=your-hf-token
```

Or add to `.env`:

```
HF_TOKEN=your-hf-token
```
