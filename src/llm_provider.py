# -*- coding: utf-8 -*-
"""
AWS_KGraph - AWS Documentation Knowledge Graph
Complete pipeline for crawling, indexing, and querying AWS documentation.
"""
from __future__ import annotations
import asyncio
import sys
import os
from pathlib import Path
from typing import List, Optional

# Import our modules
from crawler import AWSDocsCrawler, crawl_aws_docs
from rag_pipeline import RAGPipeline
from query_interface import interactive_query

def print_banner():
    """Print application banner."""
    print("=" * 60)
    print("🚀 AWS_KGRAPH - AWS KNOWLEDGE GRAPH")
    print("=" * 60)
    print("Complete toolkit for crawling, indexing, and querying AWS docs")
    print()

def print_help():
    """Print usage help."""
    print("Usage: python src/AWS_KGraph.py [COMMAND] [OPTIONS]")
    print()
    print("Commands:")
    print("  crawl     Crawl AWS documentation from web")
    print("  index     Build RAG index from crawled documents") 
    print("  query     Interactive query interface")
    print("  pipeline  Run complete pipeline: crawl + index")
    print("  demo      Run demo queries")
    print()
    print("Options:")
    print("  --services=s3,iam     Limit to specific services")
    print("  --max=N              Maximum pages to crawl")
    print("  --out=DIR            Output directory (default: aws_docs)")
    print("  --force              Force rebuild of index")
    print("  --help               Show this help")
    print()
    print("Examples:")
    print("  python src/AWS_KGraph.py crawl --services=s3,iam --max=100")
    print("  python src/AWS_KGraph.py index --force")
    print("  python src/AWS_KGraph.py pipeline --services=s3 --max=50")
    print("  python src/AWS_KGraph.py query")

def parse_args() -> dict:
    """Parse command line arguments."""
    args = {
        'command': 'help',
        'services': None,
        'max_pages': None,
        'out_dir': Path("aws_docs"),
        'force': False
    }
    
    if len(sys.argv) < 2:
        return args
    
    # Get command
    args['command'] = sys.argv[1]
    
    # Parse options
    for arg in sys.argv[2:]:
        if arg.startswith("--services="):
            services_str = arg.split("=", 1)[1]
            args['services'] = [s.strip() for s in services_str.split(",")]
        elif arg.startswith("--max="):
            try:
                args['max_pages'] = int(arg.split("=", 1)[1])
            except ValueError:
                print(f"Warning: Invalid max pages value in {arg}")
        elif arg.startswith("--out="):
            args['out_dir'] = Path(arg.split("=", 1)[1])
        elif arg == "--force":
            args['force'] = True
        elif arg == "--help":
            args['command'] = 'help'
    
    return args

async def run_crawl(services: Optional[List[str]], max_pages: Optional[int], out_dir: Path) -> bool:
    """Run the crawler."""
    print("🕷️  CRAWLING AWS DOCUMENTATION")
    print("-" * 40)
    print(f"Services: {services if services else 'all'}")
    print(f"Max pages: {max_pages if max_pages else 'unlimited'}")
    print(f"Output: {out_dir}")
    print()
    
    try:
        crawler = AWSDocsCrawler(out_dir=out_dir)
        pages_saved = await crawler.crawl(
            limit_services=services,
            max_pages=max_pages
        )
        
        print(f"\n✅ Crawling complete! Saved {pages_saved} pages to {out_dir}")
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Crawling interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Crawling failed: {e}")
        return False

async def run_index(out_dir: Path, force: bool = False) -> bool:
    """Run the indexing pipeline."""
    print("🔍 BUILDING RAG INDEX")
    print("-" * 40)
    
    # Check if docs directory exists
    if not out_dir.exists():
        print(f"❌ Documentation directory not found: {out_dir}")
        print("Please run crawl first to download AWS documentation.")
        return False
    
    # Count markdown files
    md_files = list(out_dir.rglob("*.md"))
    print(f"📁 Found {len(md_files)} markdown files in {out_dir}")
    
    if not md_files:
        print("❌ No markdown files found. Please run crawl first.")
        return False
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        return False
    
    try:
        # Initialize pipeline
        rag = RAGPipeline(docs_dir=out_dir)
        
        # Check for existing index
        if not force and rag.vector_store.load_index():
            print(f"✅ Existing index found with {rag.vector_store.index.ntotal} chunks")
            if sys.stdin.isatty():  # Only prompt if running interactively
                rebuild = input("Rebuild index? (y/N): ").strip().lower()
                force = rebuild in ['y', 'yes']
            else:
                print("Using existing index (use --force to rebuild)")
        
        if force or not rag.vector_store.load_index():
            print("🔨 Building index...")
            await rag.build_index(force_rebuild=True)
        else:
            print("✅ Using existing index")
        
        print(f"\n🎉 Index ready with {rag.vector_store.index.ntotal} document chunks!")
        rag.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to build index: {e}")
        return False

async def run_demo(out_dir: Path) -> bool:
    """Run demo queries."""
    print("🎯 RUNNING DEMO QUERIES")
    print("-" * 40)
    
    try:
        rag = RAGPipeline(docs_dir=out_dir)
        
        # Load existing index
        if not rag.vector_store.load_index():
            print("❌ No index found. Please run index command first.")
            return False
        
        print(f"✅ Loaded index with {rag.vector_store.index.ntotal} document chunks")
        
        # Demo queries
        demo_queries = [
            "What condition keys does S3 Vectors support?",
            "How do I create a vector index in S3?",
            "What are the policy actions for S3 vector buckets?",
            "How do I insert vectors into an index?",
            "What IAM permissions are needed for S3 operations?"
        ]
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n🔍 Demo Query {i}: {query}")
            print("-" * 50)
            
            try:
                results = rag.search(query, k=3)
                
                if results:
                    print("📖 Top results:")
                    for j, result in enumerate(results, 1):
                        print(f"{j}. {result['title']} (Score: {result['score']:.3f})")
                        print(f"   Service: {result['service']}")
                        print(f"   URL: {result['url']}")
                        if j == 1:  # Show snippet of top result
                            snippet = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                            print(f"   Preview: {snippet}")
                else:
                    print("❌ No results found")
                    
            except Exception as e:
                print(f"❌ Query failed: {e}")
        
        rag.close()
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

async def run_pipeline(services: Optional[List[str]], max_pages: Optional[int], out_dir: Path, force: bool = False) -> bool:
    """Run the complete pipeline: crawl + index."""
    print("🚀 RUNNING COMPLETE PIPELINE")
    print("-" * 40)
    
    success = await run_crawl(services, max_pages, out_dir)
    if not success:
        return False
    
    print("\n" + "=" * 60)
    
    # Step 2: Index
    success = await run_index(out_dir, force)
    if not success:
        return False
    
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETE!")
    print("Next steps:")
    print("• Run: python src/AWS_KGraph.py query")
    print("• Or: python src/AWS_KGraph.py demo")
    
    return True

async def main():
    """Main entry point."""
    print_banner()
    
    args = parse_args()
    
    if args['command'] == 'help':
        print_help()
        return
    
    # Validate OpenAI API key for commands that need it
    if args['command'] in ['index', 'query', 'demo', 'pipeline']:
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ OPENAI_API_KEY environment variable not set")
            print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key'")
            print("Or add it to your .env file")
            return
    
    try:
        if args['command'] == 'crawl':
            await run_crawl(args['services'], args['max_pages'], args['out_dir'])
            
        elif args['command'] == 'index':
            await run_index(args['out_dir'], args['force'])
            
        elif args['command'] == 'query':
            print("🔍 INTERACTIVE QUERY MODE")
            print("-" * 40)
            await interactive_query()
            
        elif args['command'] == 'demo':
            await run_demo(args['out_dir'])
            
        elif args['command'] == 'pipeline':
            await run_pipeline(args['services'], args['max_pages'], args['out_dir'], args['force'])
            
        else:
            print(f"❌ Unknown command: {args['command']}")
            print("Run: python src/AWS_KGraph.py help")
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Operation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())