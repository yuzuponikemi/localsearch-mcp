"""Entry point for running as a module: python -m src"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    from src.server import mcp
    from src.indexer import LocalFileIndexer
    import src.server as server_module

    # Check for build-vector-index command
    if len(sys.argv) > 1 and sys.argv[1] == "build-vector-index":
        print("Building vector index only...", file=sys.stderr)
        # For build-vector-index command, we need to create a wiki indexer instance
        from src.indexer import WikiIndexer
        wiki_idx = WikiIndexer()
        wiki_idx.load_or_build()  # Load existing BM25 index
        wiki_idx.build_vector_index()  # Build vector index from loaded documents
        print("Done!", file=sys.stderr)
    else:
        # Initialize background loading and start server
        server_module._startup_initialization()
        
        print("✅ Server ready!\n", file=sys.stderr)
        mcp.run()
