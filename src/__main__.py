"""Entry point for running as a module: python -m src"""
import sys
from .server import mcp, indexer

if __name__ == "__main__":
    print("Starting Local Wikipedia Search MCP Server...", file=sys.stderr)
    indexer.load_or_build()
    print("Server ready!\n", file=sys.stderr)
    mcp.run()
