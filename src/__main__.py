"""Entry point for running as a module: python -m src"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    from src.server import mcp, wiki_indexer
    from src.indexer import LocalFileIndexer
    import src.server as server_module

    # Check for build-vector-index command
    if len(sys.argv) > 1 and sys.argv[1] == "build-vector-index":
        print("Building vector index only...", file=sys.stderr)
        wiki_indexer.load_or_build()  # Load existing BM25 index
        wiki_indexer.build_vector_index()  # Build vector index from loaded documents
        print("Done!", file=sys.stderr)
    else:
        print("🚀 Starting Multi-Source Local Search MCP Server...", file=sys.stderr)

        # Load Wikipedia index (skip if SKIP_WIKIPEDIA is set for testing)
        skip_wikipedia = os.environ.get("SKIP_WIKIPEDIA", "false").lower() == "true"
        if not skip_wikipedia:
            print("\n📚 Loading Wikipedia index...", file=sys.stderr)
            wiki_indexer.load_or_build()
        else:
            print("\n⏭️  Skipping Wikipedia index (SKIP_WIKIPEDIA=true)", file=sys.stderr)

        # Initialize local file indexer if configured
        local_docs_path = os.environ.get("LOCAL_DOCS_PATH")
        if local_docs_path:
            print(f"\n📁 Loading local files from: {local_docs_path}", file=sys.stderr)
            try:
                server_module.local_indexer = LocalFileIndexer(local_docs_path)
                server_module.local_indexer.build_index()
            except Exception as e:
                print(f"⚠️  Warning: Failed to load local files: {e}", file=sys.stderr)
                print("   Local file search will be disabled.", file=sys.stderr)
        else:
            print("\n📁 LOCAL_DOCS_PATH not set. Local file search disabled.", file=sys.stderr)
            print("   Set LOCAL_DOCS_PATH environment variable to enable local file search.", file=sys.stderr)

        print("\n✅ Server ready!\n", file=sys.stderr)
        mcp.run()
