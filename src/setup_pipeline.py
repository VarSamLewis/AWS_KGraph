# -*- coding: utf-8 -*-
"""
Complete setup pipeline for AWS Documentation RAG
Runs crawler, chunking, embedding, and index building in one command.
"""
import asyncio
import os
import sys
from pathlib import Path

from crawler import AWSDocsCrawler
from rag_pipeline import RAGPipeline

async def main():
    """Run the complete setup pipeline."""
    print("AWS DOCS RAG SETUP PIPELINE")
    print("=" * 50)
    print("This will:")
    print("1. Check for existing AWS documentation")
    print("2. Chunk documents")
    print("3. Generate embeddings")
    print("4. Build and store vector index")
    print()
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        return False
    
    docs_dir = Path("aws_docs")
    
    try:
        # Step 1: Check for existing documentation
        print("STEP 1: CHECKING FOR AWS DOCUMENTATION")
        print("-" * 40)
        if docs_dir.exists():
            md_files = list(docs_dir.rglob("*.md"))
            print(f"Found {len(md_files)} markdown files in {docs_dir}")
            if not md_files:
                print(f"WARNING: Directory {docs_dir} exists but contains no markdown files.")
                print("You may need to run the crawler first.")
                return False
        else:
            print(f"ERROR: Documentation directory {docs_dir} does not exist.")
            print("Please run the crawler first to download AWS documentation.")
            return False
        
        # Step 2: Build RAG Index (includes chunking and embedding)
        print("\nSTEP 2: BUILDING RAG INDEX")
        print("-" * 40)
        print("This includes chunking, embedding generation, and vector store creation...")
        print()
        
        rag = RAGPipeline(docs_dir=docs_dir)
        await rag.build_index(force_rebuild=True)
        
        print()
        print("SETUP COMPLETE!")
        print("=" * 50)
        print(f"Vector index built with {rag.vector_store.index.ntotal} document chunks")
        print()
        print("Ready to query! Run: poetry run query")
        
        rag.close()
        return True
        
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        return False
    except Exception as e:
        print(f"\nSetup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)