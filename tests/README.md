# Test Suite

This directory contains different types of tests for the LocalSearch MCP Server.

## Test Types

### 1. CI/CD Tests (Automated)

**File**: `test_indexing_search.py`

Tests that run automatically in CI/CD pipelines:
- MCP server connection
- Local document indexing
- Search results quality
- Incremental indexing (mtime-based)
- Search strategies (keyword vs hybrid)

**Requirements**:
- No LLM needed
- Uses test documents in `test_docs/`
- Fast execution (< 2 minutes)

**Run**:
```bash
# Run with local files only (skips Wikipedia download)
SKIP_WIKIPEDIA=true uv run python tests/test_indexing_search.py

# Run with Wikipedia (requires internet and disk space)
uv run python tests/test_indexing_search.py
```

### 2. LLM Integration Tests (Local Only)

**File**: `verify_with_ollama.py`

Tests that require Ollama and are intended for local development only:
- Full agent workflow with function calling
- Q&A with local documents using LLM
- Multi-turn conversations
- Tool use validation

**Requirements**:
- Ollama installed and running
- Models: `llama3.2`, `command-r` (or similar)
- External test documents (not in repository)

**Run**:
```bash
# Simple MCP connection test (no LLM)
uv run python tests/verify_with_ollama.py --simple

# Local document search test (no LLM)
uv run python tests/verify_with_ollama.py --local

# Full agent test with LLM (requires Ollama)
uv run python tests/verify_with_ollama.py

# Q&A test with LLM (requires Ollama)
uv run python tests/verify_with_ollama.py --local-qa
```

## Test Data

### Repository Test Documents
- **Location**: `test_docs/`
- **Purpose**: CI/CD testing
- **Contents**: Sample documents about Python, ML, databases, web dev, etc.
- **Version controlled**: Yes

### External Test Documents
- **Location**: Set via `LOCAL_DOCS_PATH` environment variable
- **Purpose**: Local development and LLM testing
- **Contents**: Your own documents (Markdown, text files)
- **Version controlled**: No

## CI/CD Configuration

The CI/CD pipeline runs only the automated tests from `test_indexing_search.py`:
- ✅ Indexing and search functionality
- ✅ Incremental indexing
- ✅ Search result validation
- ❌ No LLM-based tests (too slow and unreliable for CI)

See `.github/workflows/test.yml` for the CI/CD configuration.

## Adding New Tests

### For CI/CD Tests
1. Add test function to `test_indexing_search.py`
2. Use only `test_docs/` for test data
3. Avoid LLM dependencies
4. Keep execution time under 30 seconds per test

### For Local LLM Tests
1. Add test function to `verify_with_ollama.py`
2. Document Ollama model requirements
3. Add clear instructions in docstrings
4. Test with your own documents

## Test Best Practices

1. **Fast**: CI tests should complete quickly (< 2 minutes total)
2. **Isolated**: Each test should be independent
3. **Deterministic**: Same input should produce same output
4. **Clear**: Use descriptive test names and log messages
5. **Documented**: Explain what each test validates
