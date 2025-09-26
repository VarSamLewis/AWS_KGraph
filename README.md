# AWS_KGraph

AWS Documentation Knowledge Graph with AI-powered querying. A complete RAG (Retrieval-Augmented Generation) pipeline for crawling, indexing, and intelligently querying AWS documentation.

## Overview

AWS_KGraph provides a comprehensive solution for working with AWS documentation through semantic search and AI-powered question answering. The system crawls AWS documentation, creates vector embeddings, and enables natural language queries with contextual responses.

## Features

- **Documentation Crawler**: Automatically crawls AWS documentation from official sitemaps
- **Intelligent Chunking**: Splits documents by headers with configurable overlap for optimal retrieval
- **Vector Embeddings**: Uses OpenAI's text-embedding-3-large model for semantic search
- **FAISS Integration**: High-performance vector similarity search with SQLite metadata storage
- **AI-Powered Queries**: Interactive CLI with GPT-4o-mini for contextual responses
- **Service Filtering**: Target specific AWS services (S3, IAM, EC2, etc.)
- **Rich CLI Interface**: Beautiful command-line interface with progress indicators and formatted output

## Installation

Ensure you have Python 3.9+ and Poetry installed.

## Architecture

### Components

- **Crawler** (`src/crawler.py`): Fetches and converts AWS documentation to Markdown
- **RAG Pipeline** (`src/rag_pipeline.py`): Core document processing and vector search
- **Query Interface** (`src/query_interface.py`): Interactive CLI for querying
- **Setup Pipeline** (`src/setup_pipeline.py`): Automated index building

### Data Flow

1. **Document Loading**: Markdown files processed with metadata extraction
2. **Text Chunking**: Content split by headers (600 tokens, 50 token overlap)
3. **Embedding Generation**: OpenAI API creates 3072-dimensional vectors
4. **Vector Storage**: FAISS index with SQLite metadata database
5. **Query Processing**: Semantic search retrieves relevant chunks for LLM context

### Storage

- `aws_docs_index.faiss`: Vector embeddings (FAISS format)
- `aws_docs_index.db`: Text content and metadata (SQLite)
- `aws_docs/`: Raw documentation files (Markdown)

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required for embeddings and chat completions

### Customization

**Chunk Settings** (in `DocumentChunker`):
- `chunk_size`: Token limit per chunk (default: 600)
- `overlap`: Overlap between chunks (default: 50)
- `min_chunk_size`: Minimum chunk size (default: 100)

**Embedding Model** (in `RAGPipeline`):
- `embedding_model`: OpenAI model (default: "text-embedding-3-large")
- `batch_size`: API batch size (default: 100)

## Cost Considerations

- **Index Building**: ~$0.68 for text-embedding-3-large (8,748 chunks)
- **Queries**: ~$0.00008 per search query
- **LLM Responses**: Variable based on context size and model (GPT-4o-mini default)

## Requirements

- Python 3.9+
- OpenAI API key
- ~200MB disk space for full AWS documentation
- ~150MB additional space for vector index

## Contributing

1. Install development dependencies: `poetry install --with dev`
2. Run tests: `poetry run pytest`
3. Format code: `poetry run black src/`
4. Sort imports: `poetry run isort src/`

## TO-DO

- [ ] Test poetry build process
- [ ] Add URL links to access docs
- [ ] Implement hybrid search (semantic + keyword)
- [ ] Add support for other embedding providers
- [ ] Add CICD for tests and build
- [ ] Implement incremental index updates

## License

MIT License - see LICENSE file for details.