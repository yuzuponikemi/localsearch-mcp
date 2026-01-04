"""Entry point for running as a module: python -m src"""
from .server import mcp, indexer

if __name__ == "__main__":
    print("Starting Local Wikipedia Search MCP Server...")
    indexer.load_or_build()
    print("Server ready!\n")
    mcp.run()
