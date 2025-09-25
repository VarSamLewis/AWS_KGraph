# -*- coding: utf-8 -*-
"""
Beautiful CLI Query Interface for AWS Documentation RAG
Interactive and command-line interface for querying AWS documentation with AI.
"""
from __future__ import annotations
import asyncio
import os
from pathlib import Path
from typing import Optional, List
import json

# CLI and UI libraries
import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich import box

# Our modules
from rag_pipeline import RAGPipeline
from llm_provider import LLMManager, LLMFactory

# Initialize Rich console and Typer app
console = Console()
app = typer.Typer(
    name="aws-docs-query",
    help="Beautiful CLI for querying AWS documentation with AI",
    add_completion=False
)

class QueryInterface:
    """Main query interface class."""
    
    def __init__(self, docs_dir: Path, llm_provider: str = "openai", llm_model: str = None):
        self.docs_dir = docs_dir
        self.rag = RAGPipeline(docs_dir=docs_dir)
        self.llm = LLMManager(provider=llm_provider, model=llm_model)
        self.session_history: List[dict] = []
    
    async def initialize(self) -> bool:
        """Initialize the RAG pipeline."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading knowledge base...", total=None)
            
            if not self.rag.vector_store.load_index():
                console.print("ERROR: No index found. Please run the indexing first.", style="red")
                console.print("TIP: Run: poetry run setup", style="yellow")
                return False
            
            progress.update(task, description="Knowledge base loaded!")
        
        return True
    
    async def ask_question(self, question: str, num_results: int = 5) -> dict:
        """Ask a question and get an AI-powered answer."""
        
        # Search for relevant context
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            search_task = progress.add_task("Searching knowledge base...", total=None)
            
            context_data = self.rag.query_with_context(question, k=num_results)
            
            progress.update(search_task, description="Generating AI response...")
            
            # Generate AI response
            llm_response = await self.llm.ask_question(
                question=question,
                context=context_data['context']
            )
            
            progress.update(search_task, description="Response generated!")
        
        result = {
            'question': question,
            'answer': llm_response.content,
            'sources': context_data['sources'],
            'model_info': {
                'provider': llm_response.provider,
                'model': llm_response.model,
                'usage': llm_response.usage
            },
            'context': context_data['context']
        }
        
        # Add to session history
        self.session_history.append(result)
        
        return result
    
    def display_response(self, result: dict, show_sources: bool = True, show_context: bool = False):
        """Display the response in a beautiful format."""
        
        # Question panel
        question_panel = Panel(
            result['question'],
            title="Question",
            border_style="blue",
            box=box.ROUNDED
        )
        console.print(question_panel)
        console.print()
        
        # Answer panel with markdown
        answer_md = Markdown(result['answer'])
        answer_panel = Panel(
            answer_md,
            title="AI Answer",
            border_style="green",
            box=box.ROUNDED
        )
        console.print(answer_panel)
        console.print()
        
        # Model info
        model_info = result['model_info']
        info_text = f"Model: {model_info['model']} | Provider: {model_info['provider']} | Tokens: {model_info['usage']['total_tokens']}"
        console.print(f"[dim]{info_text}[/dim]")
        console.print()
        
        # Sources
        if show_sources and result['sources']:
            sources_table = Table(
                title="Sources",
                box=box.ROUNDED,
                border_style="yellow"
            )
            sources_table.add_column("Title", style="cyan", no_wrap=False)
            sources_table.add_column("Service", style="magenta", justify="center")
            sources_table.add_column("Relevance", style="green", justify="center")
            sources_table.add_column("URL", style="blue", no_wrap=False)
            
            for source in result['sources']:
                sources_table.add_row(
                    source['title'],
                    source['service'].upper(),
                    f"{source['score']:.3f}",
                    f"[link={source['url']}]View docs[/link]"
                )
            
            console.print(sources_table)
            console.print()
        
        # Context (if requested)
        if show_context:
            context_panel = Panel(
                result['context'][:500] + "..." if len(result['context']) > 500 else result['context'],
                title="Context Used",
                border_style="dim",
                box=box.ROUNDED
            )
            console.print(context_panel)

@app.command()
def interactive(
    docs_dir: str = typer.Option("aws_docs", "--docs", "-d", help="Documentation directory"),
    provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider (openai, anthropic)"),
    model: str = typer.Option(None, "--model", "-m", help="Specific model to use"),
    sources: int = typer.Option(5, "--sources", "-s", help="Number of source documents to retrieve")
):
    """Start interactive question-answering session"""
    
    console.print(Panel.fit(
        "[bold green]AWS Documentation AI Assistant[/bold green]\n"
        "Ask me anything about AWS services and I'll help you find answers!",
        border_style="green"
    ))
    
    # Initialize
    interface = QueryInterface(Path(docs_dir), provider, model)
    
    if not asyncio.run(interface.initialize()):
        raise typer.Exit(1)
    
    # Show model info
    model_info = interface.llm.get_model_info()
    info_panel = Panel(
        f"Provider: {model_info['provider']}\n"
        f"Model: {model_info['model']}\n"
        f"Knowledge base: {interface.rag.vector_store.index.ntotal} document chunks",
        title="Configuration",
        border_style="blue"
    )
    console.print(info_panel)
    console.print()
    
    # Interactive loop
    try:
        while True:
            console.print("[bold yellow]" + "─" * 60 + "[/bold yellow]")
            question = Prompt.ask("Your question", console=console)
            
            if question.lower() in ['quit', 'exit', 'q', 'bye']:
                break
            
            if not question.strip():
                continue
            
            try:
                result = asyncio.run(interface.ask_question(question, sources))
                console.print()
                interface.display_response(result)
                
            except Exception as e:
                console.print(f"ERROR: {e}", style="red")
                console.print()
    
    except KeyboardInterrupt:
        pass
    
    finally:
        console.print("\nThanks for using AWS Documentation AI Assistant!")
        interface.rag.close()

@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    docs_dir: str = typer.Option("aws_docs", "--docs", "-d", help="Documentation directory"),
    provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider"),
    model: str = typer.Option(None, "--model", "-m", help="Specific model to use"),
    sources: int = typer.Option(5, "--sources", "-s", help="Number of sources to retrieve"),
    show_context: bool = typer.Option(False, "--context", help="Show context used"),
    output: str = typer.Option(None, "--output", "-o", help="Save response to file (JSON)")
):
    """Ask a single question and get an answer"""
    
    interface = QueryInterface(Path(docs_dir), provider, model)
    
    if not asyncio.run(interface.initialize()):
        raise typer.Exit(1)
    
    try:
        result = asyncio.run(interface.ask_question(question, sources))
        interface.display_response(result, show_context=show_context)
        
        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            console.print(f"\nResponse saved to {output}")
        
    except Exception as e:
        console.print(f"ERROR: {e}", style="red")
        raise typer.Exit(1)
    
    finally:
        interface.rag.close()

@app.command()
def models(provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider")):
    """List available models for a provider"""
    
    try:
        available_providers = LLMFactory.get_available_providers()
        
        if provider not in available_providers:
            console.print(f"ERROR: Unknown provider: {provider}", style="red")
            console.print(f"Available providers: {', '.join(available_providers)}")
            raise typer.Exit(1)
        
        llm = LLMManager(provider=provider)
        models = llm.provider.get_available_models()
        
        table = Table(title=f"Available Models - {provider.upper()}")
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="green")
        
        for model in models:
            table.add_row(model, "Available")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"ERROR: {e}", style="red")
        raise typer.Exit(1)

@app.command()
def demo(
    docs_dir: str = typer.Option("aws_docs", "--docs", "-d", help="Documentation directory"),
    provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider"),
    model: str = typer.Option(None, "--model", "-m", help="Specific model to use")
):
    """Run demo questions with AI responses"""
    
    demo_questions = [
        "What condition keys does S3 Vectors support?",
        "How do I create a vector index in S3?",
        "What are the policy actions for S3 vector buckets?",
        "How do I insert vectors into an index?",
        "What IAM permissions are needed for S3 operations?"
    ]
    
    interface = QueryInterface(Path(docs_dir), provider, model)
    
    if not asyncio.run(interface.initialize()):
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        "[bold green]AWS Documentation Demo[/bold green]\n"
        f"Running {len(demo_questions)} demo questions...",
        border_style="green"
    ))
    
    try:
        for i, question in enumerate(demo_questions, 1):
            console.print(f"\n[bold yellow]Demo Question {i}/{len(demo_questions)}[/bold yellow]")
            
            result = asyncio.run(interface.ask_question(question, 3))
            interface.display_response(result)
            
            if i < len(demo_questions):
                if not Confirm.ask("Continue to next question?", default=True):
                    break
    
    except KeyboardInterrupt:
        console.print("\nDemo interrupted")
    
    finally:
        interface.rag.close()

def interactive_main():
    """Entry point for poetry run query command."""
    # Run the interactive command directly
    import sys
    sys.argv = ["query_interface.py", "interactive"]
    app()

if __name__ == "__main__":
    app()