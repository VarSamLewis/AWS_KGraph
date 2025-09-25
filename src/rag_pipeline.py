# -*- coding: utf-8 -*-
"""
AWS Documentation RAG Pipeline
A complete pipeline for loading, chunking, embedding, and querying AWS documentation.
"""
from __future__ import annotations
import re
import sqlite3
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass
import tiktoken
import openai
import faiss
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag_pipeline")

@dataclass
class DocumentMetadata:
    """Metadata for a document chunk."""
    path: str
    url: str
    service: str
    filename: str
    chunk_id: int
    title: str = ""
    heading: str = ""
    token_count: int = 0
    
@dataclass
class DocumentChunk:
    """A document chunk with text and metadata."""
    text: str
    metadata: DocumentMetadata
    embedding: Optional[np.ndarray] = None

class TokenCounter:
    """Utility for counting tokens using tiktoken."""
    
    def __init__(self, model: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(model)
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

class DocumentLoader:
    """Loads and parses markdown documents from aws_docs directory."""
    
    def __init__(self, docs_dir: Path = Path("aws_docs")):
        self.docs_dir = docs_dir
        self.token_counter = TokenCounter()
    
    def load_documents(self) -> Iterator[Tuple[str, DocumentMetadata]]:
        """Load all markdown documents with metadata."""
        md_files = list(self.docs_dir.rglob("*.md"))
        log.info(f"Found {len(md_files)} markdown files")
        
        for md_file in tqdm(md_files, desc="Loading documents"):
            try:
                content = md_file.read_text(encoding="utf-8", errors="ignore")
                metadata = self._extract_metadata(md_file, content)
                if metadata:
                    clean_content = self._clean_content(content)
                    yield clean_content, metadata
            except Exception as e:
                log.warning(f"Failed to load {md_file}: {e}")
                continue
    
    def _extract_metadata(self, file_path: Path, content: str) -> Optional[DocumentMetadata]:
        """Extract metadata from markdown file."""
        try:
            # Extract URL from HTML comment header
            url_match = re.search(r"<!-- Source: (.+?) -->", content)
            if not url_match:
                log.warning(f"No source URL found in {file_path}")
                return None
            
            url = url_match.group(1).strip()
            
            # Extract service from directory structure
            parts = file_path.parts
            service = parts[-2] if len(parts) > 1 else "unknown"
            
            # Extract title from content (first heading)
            title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else file_path.stem
            
            return DocumentMetadata(
                path=str(file_path.relative_to(self.docs_dir)),
                url=url,
                service=service,
                filename=file_path.name,
                chunk_id=0,  # Will be set during chunking
                title=title
            )
        except Exception as e:
            log.warning(f"Failed to extract metadata from {file_path}: {e}")
            return None
    
    def _clean_content(self, content: str) -> str:
        """Remove HTML comment headers and clean up content."""
        # Remove HTML comment source line
        content = re.sub(r"<!-- Source: .+? -->\s*", "", content)
        
        # Remove JSON-LD schema blocks
        content = re.sub(r'\{\s*"@context".*?\}\s*', "", content, flags=re.DOTALL)
        
        # Remove navigation breadcrumbs
        content = re.sub(r"\[Documentation\].*?\[User Guide\].*?\n", "", content)
        
        # Clean up excessive whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)
        
        return content.strip()

class DocumentChunker:
    """Chunks documents into overlapping segments."""
    
    def __init__(self, 
                 chunk_size: int = 600, 
                 overlap: int = 50,
                 min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.token_counter = TokenCounter()
    
    def chunk_document(self, content: str, base_metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Chunk a document into overlapping segments."""
        chunks = []
        
        # Try to split by headers first
        header_chunks = self._split_by_headers(content)
        
        for i, (text, heading) in enumerate(header_chunks):
            token_count = self.token_counter.count_tokens(text)
            
            if token_count <= self.chunk_size:
                # Chunk fits within size limit
                metadata = DocumentMetadata(
                    path=base_metadata.path,
                    url=base_metadata.url,
                    service=base_metadata.service,
                    filename=base_metadata.filename,
                    chunk_id=len(chunks),
                    title=base_metadata.title,
                    heading=heading,
                    token_count=token_count
                )
                chunks.append(DocumentChunk(text=text, metadata=metadata))
            else:
                # Split large chunks further
                sub_chunks = self._split_large_chunk(text, heading, base_metadata, len(chunks))
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_by_headers(self, content: str) -> List[Tuple[str, str]]:
        """Split content by markdown headers."""
        # Split by headers (# ## ### etc.)
        header_pattern = r"^(#{1,6}\s+.+?)$"
        sections = []
        current_section = ""
        current_heading = ""
        
        lines = content.split('\n')
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Save previous section
                if current_section.strip():
                    sections.append((current_section.strip(), current_heading))
                
                # Start new section
                current_heading = header_match.group(1)
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        # Add final section
        if current_section.strip():
            sections.append((current_section.strip(), current_heading))
        
        # If no headers found, treat as single section
        if not sections:
            sections.append((content, ""))
        
        return sections
    
    def _split_large_chunk(self, text: str, heading: str, base_metadata: DocumentMetadata, chunk_offset: int) -> List[DocumentChunk]:
        """Split a large chunk into smaller overlapping pieces."""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk
                if current_tokens >= self.min_chunk_size:
                    metadata = DocumentMetadata(
                        path=base_metadata.path,
                        url=base_metadata.url,
                        service=base_metadata.service,
                        filename=base_metadata.filename,
                        chunk_id=chunk_offset + len(chunks),
                        title=base_metadata.title,
                        heading=heading,
                        token_count=current_tokens
                    )
                    chunks.append(DocumentChunk(text=current_chunk.strip(), metadata=metadata))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.overlap)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.token_counter.count_tokens(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk.strip() and current_tokens >= self.min_chunk_size:
            metadata = DocumentMetadata(
                path=base_metadata.path,
                url=base_metadata.url,
                service=base_metadata.service,
                filename=base_metadata.filename,
                chunk_id=chunk_offset + len(chunks),
                title=base_metadata.title,
                heading=heading,
                token_count=current_tokens
            )
            chunks.append(DocumentChunk(text=current_chunk.strip(), metadata=metadata))
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last N tokens from text for overlap."""
        tokens = self.token_counter.encoding.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        
        overlap_token_ids = tokens[-overlap_tokens:]
        return self.token_counter.encoding.decode(overlap_token_ids)

class EmbeddingGenerator:
    """Generates embeddings using OpenAI API."""
    
    def __init__(self, 
                 model: str = "text-embedding-3-large",
                 api_key: Optional[str] = None,
                 batch_size: int = 100):
        self.model = model
        self.batch_size = batch_size
        self.client = openai.OpenAI(api_key=api_key)
        self.embedding_dim = 3072 if "large" in model else 1536
    
    async def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for document chunks."""
        log.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Process in batches to avoid rate limits
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Generating embeddings"):
            batch = chunks[i:i + self.batch_size]
            texts = [chunk.text for chunk in batch]
            
            try:
                # Generate embeddings for batch
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )
                
                # Assign embeddings to chunks
                for j, chunk in enumerate(batch):
                    embedding = response.data[j].embedding
                    chunk.embedding = np.array(embedding, dtype=np.float32)
                    
            except Exception as e:
                log.error(f"Failed to generate embeddings for batch {i//self.batch_size}: {e}")
                # Assign zero embeddings as fallback
                for chunk in batch:
                    chunk.embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        return chunks

class VectorStore:
    """FAISS-based vector store with SQLite metadata."""
    
    def __init__(self, index_path: str = "aws_docs_index", embedding_dim: int = 3072):
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.index_file = f"{index_path}.faiss"
        self.metadata_file = f"{index_path}.db"
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        self.chunk_count = 0
        
        # Initialize SQLite for metadata
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metadata."""
        self.conn = sqlite3.connect(self.metadata_file)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                path TEXT,
                url TEXT,
                service TEXT,
                filename TEXT,
                chunk_id INTEGER,
                title TEXT,
                heading TEXT,
                token_count INTEGER,
                text TEXT
            )
        """)
        self.conn.commit()
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add document chunks to the vector store."""
        if not chunks:
            return
        
        log.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Prepare embeddings and metadata
        embeddings = []
        metadata_rows = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                log.warning(f"Chunk missing embedding: {chunk.metadata.path}:{chunk.metadata.chunk_id}")
                continue
            
            # Normalize embedding for cosine similarity
            embedding = chunk.embedding / np.linalg.norm(chunk.embedding)
            embeddings.append(embedding)
            
            metadata_rows.append((
                self.chunk_count,
                chunk.metadata.path,
                chunk.metadata.url,
                chunk.metadata.service,
                chunk.metadata.filename,
                chunk.metadata.chunk_id,
                chunk.metadata.title,
                chunk.metadata.heading,
                chunk.metadata.token_count,
                chunk.text
            ))
            self.chunk_count += 1
        
        if embeddings:
            # Add to FAISS index
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings_array)
            
            # Add metadata to SQLite
            self.conn.executemany("""
                INSERT INTO chunks 
                (id, path, url, service, filename, chunk_id, title, heading, token_count, text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, metadata_rows)
            self.conn.commit()
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        if self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, k)
        
        # Retrieve metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
                
            cursor = self.conn.execute("""
                SELECT path, url, service, filename, chunk_id, title, heading, token_count, text
                FROM chunks WHERE id = ?
            """, (int(idx),))
            
            row = cursor.fetchone()
            if row:
                results.append({
                    'score': float(score),
                    'path': row[0],
                    'url': row[1],
                    'service': row[2],
                    'filename': row[3],
                    'chunk_id': row[4],
                    'title': row[5],
                    'heading': row[6],
                    'token_count': row[7],
                    'text': row[8]
                })
        
        return results
    
    def save_index(self):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, self.index_file)
        log.info(f"Saved index with {self.index.ntotal} vectors to {self.index_file}")
    
    def load_index(self) -> bool:
        """Load FAISS index from disk."""
        try:
            if Path(self.index_file).exists() and Path(self.metadata_file).exists():
                self.index = faiss.read_index(self.index_file)
                self.chunk_count = self.index.ntotal
                log.info(f"Loaded index with {self.index.ntotal} vectors from {self.index_file}")
                return True
        except Exception as e:
            log.warning(f"Failed to load index: {e}")
        
        return False
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()

class RAGPipeline:
    """Complete RAG pipeline for AWS documentation."""
    
    def __init__(self, 
                 docs_dir: Path = Path("aws_docs"),
                 index_path: str = "aws_docs_index",
                 embedding_model: str = "text-embedding-3-large",
                 openai_api_key: Optional[str] = None):
        self.docs_dir = docs_dir
        self.loader = DocumentLoader(docs_dir)
        self.chunker = DocumentChunker()
        self.embedder = EmbeddingGenerator(embedding_model, openai_api_key)
        self.vector_store = VectorStore(index_path, self.embedder.embedding_dim)
    
    async def build_index(self, force_rebuild: bool = False):
        # Try to load existing index
        if not force_rebuild and self.vector_store.load_index():
            log.info("Loaded existing index")
            return
        
        log.info("Building index from scratch...")
        
        all_chunks = []
        
        # Load and chunk documents
        for content, metadata in self.loader.load_documents():
            chunks = self.chunker.chunk_document(content, metadata)
            all_chunks.extend(chunks)
        
        log.info(f"Created {len(all_chunks)} chunks from documents")
        
        # Generate embeddings
        chunks_with_embeddings = await self.embedder.generate_embeddings(all_chunks)
        
        # Add to vector store
        self.vector_store.add_chunks(chunks_with_embeddings)
        
        # Save index
        self.vector_store.save_index()
        
        log.info("Index building complete!")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        # Generate query embedding
        response = self.embedder.client.embeddings.create(
            input=[query],
            model=self.embedder.model
        )
        query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k)
        
        return results
    
    def query_with_context(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Query with full context for LLM."""
        results = self.search(question, k)
        
        # Build context
        context_parts = []
        sources = []
        
        for result in results:
            context_parts.append(f"**{result['title']}** ({result['service']})\n{result['text']}")
            sources.append({
                'url': result['url'],
                'title': result['title'],
                'service': result['service'],
                'score': result['score']
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        system_message = """You are an AWS documentation expert. Answer the user's question using ONLY the provided AWS documentation context. 

Rules:
1. Only use information from the provided context
2. Always cite the source URL for any claims
3. If the context doesn't contain enough information, say so
4. Be precise and technical when appropriate
5. Format your response clearly with headings and bullet points when helpful

Context:
{context}
"""
        
        return {
            'question': question,
            'context': context,
            'sources': sources,
            'system_message': system_message.format(context=context)
        }
    
    def close(self):
        """Clean up resources."""
        self.vector_store.close()

# Example usage and demonstration
async def main():
    """Demonstration of the RAG pipeline."""
    # Initialize pipeline
    rag = RAGPipeline()
    
    try:
        # Build or load index
        await rag.build_index()
        
        # Example queries
        test_queries = [
            "What condition keys does S3 Vectors support?",
            "How do I create a vector index in S3?",
            "What are the policy actions for S3 vector buckets?",
            "How do I insert vectors into an index?",
            "What embedding models work with S3 Vectors?"
        ]
        
        print("\n" + "="*80)
        print("AWS DOCUMENTATION RAG PIPELINE DEMO")
        print("="*80)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 60)
            
            # Get results with context
            result = rag.query_with_context(query, k=3)
            
            print("\nSources found:")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['title']} (Score: {source['score']:.3f})")
                print(f"   Service: {source['service']}")
                print(f"   URL: {source['url']}")
            
            print(f"\nSample context (first 300 chars):")
            print(result['context'][:300] + "..." if len(result['context']) > 300 else result['context'])
            
            print("\n" + "="*80)
    
    finally:
        rag.close()

if __name__ == "__main__":
    # Set your OpenAI API key as environment variable or pass directly
    # export OPENAI_API_KEY="your-key-here"
    asyncio.run(main())