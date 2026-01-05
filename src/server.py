"""
Multi-Source Local Search MCP Server
Provides offline search via MCP protocol combining:
- Wikipedia (static, large-scale knowledge)
- Local Files (dynamic, personal knowledge)
Supports hybrid search: BM25 (keyword) + Vector (semantic) search.
"""
import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from src.indexer import WikiIndexer, LocalFileIndexer

# Load environment variables from .env file
load_dotenv()

# Initialize MCP server
mcp = FastMCP("MultiSourceLocalSearch")

# Global indexer instances
wiki_indexer = WikiIndexer()
local_indexer = None  # Will be initialized if LOCAL_DOCS_PATH is set


@mcp.resource("config://status")
def get_status() -> str:
    """Check if the search indices are loaded and ready."""
    status_lines = []

    # Wikipedia status
    if wiki_indexer.bm25:
        status_lines.append(f"📚 Wikipedia: {len(wiki_indexer.documents)} documents indexed")
    else:
        status_lines.append("📚 Wikipedia: Not loaded")

    # Local files status
    if local_indexer and local_indexer.bm25:
        status_lines.append(f"📁 Local Files: {len(local_indexer.documents)} files indexed")
    else:
        status_lines.append("📁 Local Files: Not configured or empty")

    return "\n".join(status_lines)


@mcp.tool()
def search(
    query: str,
    top_k: int = 5,
    strategy: str = "hybrid",
    source: str = "all"
) -> str:
    """
    Multi-source hybrid search across Wikipedia and local files.

    Search through locally indexed Wikipedia articles and your personal files
    using both keyword matching (BM25) and semantic similarity (vector embeddings).
    Results are intelligently merged using Reciprocal Rank Fusion (RRF).

    Args:
        query: The search keywords or question (e.g., "machine learning algorithms")
        top_k: Number of results to return per source (default: 5, max: 20)
        strategy: Search strategy - 'hybrid' (default, best results), 'keyword' (BM25 only), or 'semantic' (vector only)
        source: Data source - 'all' (default), 'wikipedia', or 'local'

    Returns:
        Formatted search results with titles, sources, and content snippets

    Examples:
        - source='all': Search both Wikipedia and local files (best for comprehensive results)
        - source='wikipedia': Search only Wikipedia (good for general knowledge)
        - source='local': Search only local files (good for personal notes/documents)
        - strategy='hybrid': Combines keyword and semantic search for best results
        - strategy='keyword': Traditional keyword-based search (fast, exact matches)
        - strategy='semantic': Meaning-based search (finds similar concepts)
    """
    # Validate and limit top_k
    top_k = min(max(1, top_k), 20)

    # Validate strategy
    if strategy not in ["keyword", "semantic", "hybrid"]:
        strategy = "hybrid"

    # Validate source
    if source not in ["all", "wikipedia", "local"]:
        source = "all"

    all_results = []

    # Search Wikipedia
    if source in ["all", "wikipedia"]:
        if not wiki_indexer.bm25:
            wiki_indexer.load_or_build()

        wiki_results = wiki_indexer.hybrid_search(query, top_k=top_k, strategy=strategy)
        for doc in wiki_results:
            doc['data_source'] = 'Wikipedia'
        all_results.extend(wiki_results)

    # Search Local Files
    if source in ["all", "local"]:
        if local_indexer and local_indexer.documents:
            local_results = local_indexer.hybrid_search(query, top_k=top_k, strategy=strategy)
            for doc in local_results:
                doc['data_source'] = 'Local Files'
            all_results.extend(local_results)
        elif source == "local":
            return "Local file search is not configured. Set LOCAL_DOCS_PATH environment variable."

    if not all_results:
        return "No results found. Try rephrasing your query or using different keywords."

    # Format results for readability
    formatted_results = []
    for i, doc in enumerate(all_results, 1):
        search_method = doc.get('source', 'unknown')
        data_source = doc.get('data_source', 'Unknown')

        formatted_results.append(
            f"[Result {i}] ({data_source} / {search_method})\n"
            f"Title: {doc['title']}\n"
            f"URL: {doc['url']}\n"
            f"Content: {doc['text']}\n"
        )

    return "\n---\n".join(formatted_results)


@mcp.tool()
def search_wikipedia(query: str, top_k: int = 3, strategy: str = "hybrid") -> str:
    """
    Search English Wikipedia for a given query using hybrid search (BM25 + Vector).

    This tool searches through locally indexed Wikipedia articles and returns
    the most relevant results using both keyword matching (BM25) and semantic
    similarity (vector embeddings). Use this when you need to find facts, history,
    definitions, or general knowledge about any topic.

    Args:
        query: The search keywords or question (e.g., "history of python programming")
        top_k: Number of results to return (default: 3, max: 10)
        strategy: Search strategy - 'hybrid' (default, best results), 'keyword' (BM25 only), or 'semantic' (vector only)

    Returns:
        Formatted search results with titles, sources, and content snippets

    Examples:
        - strategy='hybrid': Combines keyword and semantic search for best results
        - strategy='keyword': Traditional keyword-based search (fast, exact matches)
        - strategy='semantic': Meaning-based search (finds similar concepts even without exact words)
    """
    # Use the unified search function with source='wikipedia'
    return search(query=query, top_k=top_k, strategy=strategy, source="wikipedia")


@mcp.tool()
def search_local(query: str, top_k: int = 5, strategy: str = "hybrid") -> str:
    """
    Search your local files (Markdown, text) using hybrid search (BM25 + Vector).

    This tool searches through your personal files and documents using both
    keyword matching (BM25) and semantic similarity (vector embeddings).
    Perfect for finding information in your Obsidian vault, notes, or documentation.

    Args:
        query: The search keywords or question (e.g., "project meeting notes")
        top_k: Number of results to return (default: 5, max: 20)
        strategy: Search strategy - 'hybrid' (default, best results), 'keyword' (BM25 only), or 'semantic' (vector only)

    Returns:
        Formatted search results with titles, file paths, and content snippets

    Examples:
        - strategy='hybrid': Combines keyword and semantic search for best results
        - strategy='keyword': Traditional keyword-based search (fast, exact matches)
        - strategy='semantic': Meaning-based search (finds similar concepts)
    """
    # Use the unified search function with source='local'
    return search(query=query, top_k=top_k, strategy=strategy, source="local")


if __name__ == "__main__":
    import sys

    print("🚀 Starting Multi-Source Local Search MCP Server...", file=sys.stderr)

    # Load Wikipedia index
    print("\n📚 Loading Wikipedia index...", file=sys.stderr)
    wiki_indexer.load_or_build()

    # Initialize local file indexer if configured
    local_docs_path = os.environ.get("LOCAL_DOCS_PATH")
    if local_docs_path:
        print(f"\n📁 Loading local files from: {local_docs_path}", file=sys.stderr)
        try:
            local_indexer = LocalFileIndexer(local_docs_path)
            local_indexer.build_index()
        except Exception as e:
            print(f"⚠️  Warning: Failed to load local files: {e}", file=sys.stderr)
            print("   Local file search will be disabled.", file=sys.stderr)
    else:
        print("\n📁 LOCAL_DOCS_PATH not set. Local file search disabled.", file=sys.stderr)
        print("   Set LOCAL_DOCS_PATH environment variable to enable local file search.", file=sys.stderr)

    print("\n✅ Server ready!\n", file=sys.stderr)

    # Start the MCP server
    mcp.run()
