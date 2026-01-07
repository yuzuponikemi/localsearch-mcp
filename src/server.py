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
local_indexer = None  # Will be initialized on first use if LOCAL_DOCS_PATH is set
_local_docs_path_cached = None


def _ensure_local_indexer():
    """Initialize local indexer on first use (lazy initialization)."""
    global local_indexer, _local_docs_path_cached

    local_docs_path = os.environ.get("LOCAL_DOCS_PATH")

    # Check if we need to create a new indexer (path changed or first time)
    if local_indexer is not None and _local_docs_path_cached == local_docs_path:
        return  # Already initialized with the same path

    _local_docs_path_cached = local_docs_path

    if local_docs_path:
        print(f"📁 Loading local files from: {local_docs_path}", file=os.sys.stderr)
        try:
            local_indexer = LocalFileIndexer(local_docs_path)
            local_indexer.build_index()
        except Exception as e:
            print(f"⚠️  Warning: Failed to load local files: {e}", file=os.sys.stderr)
            print("   Local file search will be disabled.", file=os.sys.stderr)
            local_indexer = None


def _ensure_wiki_indexer():
    """Initialize Wikipedia indexer on first use (lazy initialization)."""
    global wiki_indexer

    skip_wiki = os.environ.get("SKIP_WIKIPEDIA", "").lower() == "true"

    if skip_wiki:
        print("⏭️  Skipping Wikipedia index (SKIP_WIKIPEDIA=true)", file=os.sys.stderr)
        return

    if not wiki_indexer.bm25:
        print("📚 Loading Wikipedia index...", file=os.sys.stderr)
        wiki_indexer.load_or_build()


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
def query_internal_knowledge_base(
    query: str,
    top_k: int = 5,
    strategy: str = "hybrid",
    source: str = "all"
) -> str:
    """
    Search internal confidential technical documents and proprietary knowledge bases.

    === CONTENT (What this tool searches) ===
    This tool searches through INTERNAL, NON-PUBLIC sources including:
    - Confidential design documents and architecture specifications
    - Legacy hardware specifications and datasheets (e.g., custom ASIC designs, FPGA configs)
    - Proprietary technology documentation and internal standards
    - Company-specific regulations, guidelines, and best practices
    - Internal meeting notes, project documentation, and technical memos
    - Locally indexed Wikipedia articles (for offline general knowledge)

    === WHEN TO USE THIS TOOL (Triggers) ===
    PRIORITIZE this tool when the user asks about:
    - "社内規定" (internal regulations), "設計書" (design documents)
    - Past design decisions, legacy system specifications
    - Hardware specifications with specific model numbers (e.g., "HW-2024-A", "CTRL-X100")
    - Internal project codenames or proprietary terminology
    - Questions that require company-specific knowledge NOT available on the public internet
    - Offline search when internet access is unavailable

    === WHEN NOT TO USE THIS TOOL (Negative Constraints) ===
    DO NOT use this tool for:
    - General Python/JavaScript coding questions (use web search or built-in knowledge)
    - Publicly available library documentation (e.g., React, Django, NumPy)
    - Current events, news, or real-time information
    - Generic technical questions answerable via public Stack Overflow or documentation

    Args:
        query: The search keywords or question (e.g., "HW-2024-A電源仕様", "認証モジュール設計")
        top_k: Number of results to return per source (default: 5, max: 20)
        strategy: Search strategy - 'hybrid' (default, best results), 'keyword' (BM25 only), or 'semantic' (vector only)
        source: Data source - 'all' (default), 'wikipedia', or 'local'

    Returns:
        Formatted search results with titles, sources, and content snippets
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
        _ensure_wiki_indexer()

        if wiki_indexer.bm25:
            wiki_results = wiki_indexer.hybrid_search(query, top_k=top_k, strategy=strategy)
            for doc in wiki_results:
                doc['data_source'] = 'Wikipedia'
            all_results.extend(wiki_results)

    # Search Local Files
    if source in ["all", "local"]:
        _ensure_local_indexer()

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
def search_offline_wikipedia(query: str, top_k: int = 3, strategy: str = "hybrid") -> str:
    """
    Search OFFLINE locally-indexed Wikipedia for general knowledge (no internet required).

    === CONTENT ===
    Pre-indexed English Wikipedia articles stored locally. Useful for:
    - Historical facts, scientific concepts, definitions
    - Background knowledge to supplement internal document searches
    - Offline environments where web search is unavailable

    === WHEN TO USE ===
    - User needs general encyclopedic knowledge AND is working offline
    - Supplementing internal document searches with public background info
    - Internet connectivity is restricted or unavailable

    === WHEN NOT TO USE ===
    - For internal/confidential company information (use search_internal_technical_documents)
    - When up-to-date information is needed (this index may be outdated)
    - When web search is available and preferred

    Args:
        query: The search keywords or question (e.g., "history of python programming")
        top_k: Number of results to return (default: 3, max: 10)
        strategy: Search strategy - 'hybrid' (default), 'keyword' (BM25 only), or 'semantic' (vector only)

    Returns:
        Formatted search results with titles, sources, and content snippets
    """
    # Use the unified search function with source='wikipedia'
    return query_internal_knowledge_base(query=query, top_k=top_k, strategy=strategy, source="wikipedia")


@mcp.tool()
def search_internal_technical_documents(query: str, top_k: int = 5, strategy: str = "hybrid") -> str:
    """
    Search internal confidential design documents, hardware specs, and proprietary technical knowledge.

    === CONTENT (What this tool searches) ===
    This tool provides access to INTERNAL, CONFIDENTIAL company resources:
    - 機密設計書 (Confidential design documents and architecture specs)
    - レガシーハードウェア仕様書 (Legacy hardware specifications, e.g., HW-2024-A, CTRL-X100, PWR-500)
    - 独自技術ドキュメント (Proprietary technology documentation)
    - 社内規定・ガイドライン (Internal regulations and guidelines)
    - プロジェクト議事録・技術メモ (Project meeting notes and technical memos)

    === WHEN TO USE THIS TOOL (Triggers) - PRIORITIZE THIS TOOL ===
    Use this tool FIRST when the user asks about:
    - 「社内規定」「内部ガイドライン」(internal regulations/guidelines)
    - 「過去の設計」「レガシーシステム」(past designs, legacy systems)
    - Hardware specs with model numbers: "HW-2024-A仕様", "CTRL-X100 pinout", "PWR-500電源設計"
    - Internal project codenames (e.g., "Project Phoenix", "Eagle認証モジュール")
    - Company-specific terminology or processes not found on public internet
    - 「〇〇の設計書どこ？」「△△の仕様教えて」type questions

    === WHEN NOT TO USE THIS TOOL (Negative Constraints) ===
    DO NOT use this tool for:
    ✗ General Python/JavaScript/TypeScript coding questions
    ✗ Public library docs (React, Django, NumPy, TensorFlow, etc.)
    ✗ Stack Overflow-type questions with publicly available answers
    ✗ Current events, news, or real-time market data
    ✗ Generic "how to" programming tutorials

    Args:
        query: Search terms (e.g., "HW-2024-A電源仕様", "認証フロー設計", "プロジェクトX要件定義")
        top_k: Number of results to return (default: 5, max: 20)
        strategy: 'hybrid' (default, recommended), 'keyword' (exact match), or 'semantic' (concept match)

    Returns:
        Formatted search results with document titles, file paths, and content excerpts
    """
    # Use the unified search function with source='local'
    return query_internal_knowledge_base(query=query, top_k=top_k, strategy=strategy, source="local")


if __name__ == "__main__":
    import sys

    print("🚀 Starting Multi-Source Local Search MCP Server...", file=sys.stderr)

    # Start the MCP server (initialization will happen in on_startup hook)
    mcp.run()
