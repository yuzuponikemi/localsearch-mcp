# API Usage Guide

This guide explains how to use the LocalSearch MCP Server API through the Model Context Protocol.

## Available Tools

### 1. search_wikipedia

Search through Wikipedia articles using hybrid search (BM25 + vector embeddings).

**Parameters:**
- `query` (string, required): The search query
- `top_k` (integer, optional): Number of results to return (default: 5)

**Example:**
```json
{
  "query": "quantum computing",
  "top_k": 3
}
```

### 2. search_local

Search through your local documents (Markdown and text files).

**Parameters:**
- `query` (string, required): The search query
- `top_k` (integer, optional): Number of results to return (default: 5)
- `strategy` (string, optional): Search strategy - "hybrid" or "keyword" (default: "hybrid")

**Example:**
```json
{
  "query": "machine learning algorithms",
  "top_k": 5,
  "strategy": "hybrid"
}
```

### 3. search

Multi-source search across both Wikipedia and local files.

**Parameters:**
- `query` (string, required): The search query
- `top_k` (integer, optional): Number of results per source (default: 5)
- `source` (string, optional): Search source - "wikipedia", "local", or "all" (default: "all")

**Example:**
```json
{
  "query": "neural networks",
  "top_k": 3,
  "source": "all"
}
```

## Response Format

All search tools return results in the following format:

```
Source: [Wikipedia/Local]
Title: [Document Title]
Score: [Relevance Score]
Content: [Document Excerpt]
---
```

## Best Practices

1. **Query Formulation**: Use specific keywords for better results
2. **Top K Selection**: Start with 3-5 results, increase if needed
3. **Strategy Selection**: Use "hybrid" for semantic search, "keyword" for exact matches
4. **Error Handling**: Check for empty results and handle appropriately

## Integration Examples

### With Ollama
See `tests/verify_with_ollama.py` for a complete example of using this MCP server with Ollama for function calling.

### With Claude Desktop
Add to your Claude Desktop configuration:
```json
{
  "mcpServers": {
    "localsearch": {
      "command": "uv",
      "args": ["run", "python", "-m", "src"],
      "env": {
        "LOCAL_DOCS_PATH": "/path/to/your/documents"
      }
    }
  }
}
```

## Troubleshooting

**Issue**: No results returned
- Check if Wikipedia index is built
- Verify LOCAL_DOCS_PATH is set correctly
- Ensure documents exist in the specified path

**Issue**: Slow search performance
- Reduce top_k value
- Use "keyword" strategy instead of "hybrid"
- Check system resources (RAM, CPU)

For more information, see the main README.md file.
