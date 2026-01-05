"""
Local Wikipedia Search MCP Server
Provides offline Wikipedia search via MCP protocol using BM25 algorithm.
"""
from mcp.server.fastmcp import FastMCP
try:
    from .indexer import WikiIndexer
except ImportError:
    from indexer import WikiIndexer

# Initialize MCP server
mcp = FastMCP("LocalWikiSearch")

# Global indexer instance
indexer = WikiIndexer()


@mcp.resource("config://status")
def get_status() -> str:
    """Check if the search index is loaded and ready."""
    if indexer.bm25:
        return f"Index Loaded - {len(indexer.documents)} documents available"
    return "Index Not Loaded"


@mcp.tool()
def search_wikipedia(query: str, top_k: int = 3) -> str:
    """
    Search English Wikipedia for a given query using BM25 algorithm.

    This tool searches through locally indexed Wikipedia articles and returns
    the most relevant results. Use this when you need to find facts, history,
    definitions, or general knowledge about any topic.

    Args:
        query: The search keywords (e.g., "history of python programming")
        top_k: Number of results to return (default: 3, max: 10)

    Returns:
        Formatted search results with titles, sources, and content snippets
    """
    # Ensure index is loaded (lazy loading on first call)
    if not indexer.bm25:
        indexer.load_or_build()

    # Validate and limit top_k
    top_k = min(max(1, top_k), 10)

    # Perform search
    results = indexer.search(query, top_k=top_k)

    if not results:
        return "No results found for your query. Try rephrasing or using different keywords."

    # Format results for readability
    formatted_results = []
    for i, doc in enumerate(results, 1):
        formatted_results.append(
            f"[Result {i}]\n"
            f"Title: {doc['title']}\n"
            f"Source: {doc['url']}\n"
            f"Content: {doc['text']}\n"
        )

    return "\n---\n".join(formatted_results)


if __name__ == "__main__":
    import sys
    # Pre-load index before starting server
    print("🚀 Starting Local Wikipedia Search MCP Server...", file=sys.stderr)
    indexer.load_or_build()
    print("✅ Server ready!\n", file=sys.stderr)

    # Start the MCP server
    mcp.run()
