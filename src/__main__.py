"""Entry point for running as a module: python -m src"""
import sys
from .server import mcp, indexer

if __name__ == "__main__":
    # Check for build-vector-index command
    if len(sys.argv) > 1 and sys.argv[1] == "build-vector-index":
        print("Building vector index only...", file=sys.stderr)
        indexer.load_or_build()  # Load existing BM25 index
        indexer.build_vector_index()  # Build vector index from loaded documents
        print("Done!", file=sys.stderr)
    else:
        print("Starting Local Wikipedia Search MCP Server...", file=sys.stderr)
        indexer.load_or_build()
        print("Server ready!\n", file=sys.stderr)
        mcp.run()
