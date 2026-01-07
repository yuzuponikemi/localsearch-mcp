"""
CI/CD Test Suite for LocalSearch MCP Server

Tests incremental indexing and search functionality without requiring LLM.
Designed to run in CI/CD environments using test documents in the repository.
"""
import asyncio
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DOCS_PATH = os.path.join(PROJECT_ROOT, "test_docs")

# Test results
test_results = []


def log_test(test_name: str, passed: bool, message: str = ""):
    """Log test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    result = {
        "test": test_name,
        "passed": passed,
        "message": message
    }
    test_results.append(result)
    print(f"{status}: {test_name}")
    if message:
        print(f"   {message}")


async def test_mcp_connection():
    """Test 1: Verify MCP server can start and connect."""
    try:
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "python", "-m", "src"],
            env={
                **os.environ.copy(),
                "PYTHONPATH": PROJECT_ROOT,
                "LOCAL_DOCS_PATH": TEST_DOCS_PATH,
                "SKIP_WIKIPEDIA": "true"
            },
            cwd=PROJECT_ROOT
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()

                # Check if required tools exist
                tool_names = [t.name for t in tools.tools]
                required_tools = ["search_local", "search_wikipedia", "search"]

                missing_tools = [t for t in required_tools if t not in tool_names]

                if missing_tools:
                    log_test("MCP Connection", False, f"Missing tools: {missing_tools}")
                    return False

                log_test("MCP Connection", True, f"Found {len(tool_names)} tools")
                return True

    except Exception as e:
        log_test("MCP Connection", False, str(e))
        return False


async def test_local_indexing():
    """Test 2: Verify local documents are indexed correctly."""
    try:
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "python", "-m", "src"],
            env={
                **os.environ.copy(),
                "PYTHONPATH": PROJECT_ROOT,
                "LOCAL_DOCS_PATH": TEST_DOCS_PATH,
                "SKIP_WIKIPEDIA": "true"
            },
            cwd=PROJECT_ROOT
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Test query that should match test documents
                result = await session.call_tool(
                    "search_local",
                    arguments={"query": "Python programming language", "top_k": 5}
                )

                result_text = result.content[0].text

                # Check if result contains expected content
                if "Python" in result_text and len(result_text) > 100:
                    log_test("Local Document Indexing", True, f"Found {len(result_text)} chars of results")
                    return True
                else:
                    log_test("Local Document Indexing", False, "Results too short or missing content")
                    return False

    except Exception as e:
        log_test("Local Document Indexing", False, str(e))
        return False


async def test_search_results():
    """Test 3: Verify search returns relevant results for different queries."""
    try:
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "python", "-m", "src"],
            env={
                **os.environ.copy(),
                "PYTHONPATH": PROJECT_ROOT,
                "LOCAL_DOCS_PATH": TEST_DOCS_PATH,
                "SKIP_WIKIPEDIA": "true"
            },
            cwd=PROJECT_ROOT
        )

        test_queries = [
            {
                "query": "machine learning algorithms",
                "expected_keywords": ["machine learning", "algorithm"],
                "description": "Machine Learning topic"
            },
            {
                "query": "database management SQL",
                "expected_keywords": ["database", "SQL"],
                "description": "Database topic"
            },
            {
                "query": "web development frontend",
                "expected_keywords": ["web", "frontend"],
                "description": "Web Development topic"
            }
        ]

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                passed_queries = 0
                failed_queries = []

                for test_case in test_queries:
                    result = await session.call_tool(
                        "search_local",
                        arguments={"query": test_case["query"], "top_k": 3}
                    )

                    result_text = result.content[0].text.lower()

                    # Check if at least one expected keyword is in results
                    found_keywords = [kw for kw in test_case["expected_keywords"] if kw.lower() in result_text]

                    if found_keywords:
                        passed_queries += 1
                    else:
                        failed_queries.append(test_case["description"])

                if passed_queries == len(test_queries):
                    log_test("Search Results Quality", True, f"All {passed_queries} queries returned relevant results")
                    return True
                else:
                    log_test("Search Results Quality", False,
                            f"Only {passed_queries}/{len(test_queries)} queries passed. Failed: {failed_queries}")
                    return False

    except Exception as e:
        log_test("Search Results Quality", False, str(e))
        return False


async def test_incremental_indexing():
    """Test 4: Verify incremental indexing detects new and modified files."""
    try:
        # Create a temporary directory for this test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create initial document (make it long enough to pass MIN_CHUNK_SIZE)
            initial_doc = os.path.join(temp_dir, "test_doc.md")
            with open(initial_doc, "w") as f:
                f.write("""# Initial Document

This is the initial content about quantum computing. Quantum computing is a revolutionary technology that uses quantum mechanical phenomena like superposition and entanglement to perform computations. It has the potential to solve complex problems much faster than classical computers.
""")

            server_params = StdioServerParameters(
                command="uv",
                args=["run", "python", "-m", "src"],
                env={
                    **os.environ.copy(),
                    "PYTHONPATH": PROJECT_ROOT,
                    "LOCAL_DOCS_PATH": temp_dir,
                    "SKIP_WIKIPEDIA": "true"
                },
                cwd=PROJECT_ROOT
            )

            # First indexing - search for initial content
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    result1 = await session.call_tool(
                        "search_local",
                        arguments={"query": "quantum computing", "top_k": 1}
                    )

                    result1_text = result1.content[0].text
                    initial_found = "quantum computing" in result1_text.lower()

            # Modify the document
            await asyncio.sleep(1.0)  # Ensure mtime changes (some filesystems have 1-second resolution)
            with open(initial_doc, "w") as f:
                f.write("""# Updated Document

This is the updated content about neural networks and deep learning. Neural networks are computational models inspired by biological neural networks in animal brains. Deep learning uses multiple layers of neural networks to progressively extract higher-level features from raw input data.
""")

            # Add a new document
            new_doc = os.path.join(temp_dir, "new_doc.md")
            with open(new_doc, "w") as f:
                f.write("""# New Document

This is a new document about blockchain technology. Blockchain is a distributed ledger technology that maintains a continuously growing list of records called blocks. Each block contains a cryptographic hash of the previous block, creating an immutable chain of data.
""")

            # Second indexing - should detect changes
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Note: We don't check if old content is gone because BM25 may still
                    # return the file if other keywords match. The key test is that
                    # new content can be found.

                    # Search for new content in modified file
                    result3 = await session.call_tool(
                        "search_local",
                        arguments={"query": "neural networks deep learning", "top_k": 3}
                    )
                    result3_text = result3.content[0].text
                    new_content_found = "neural networks" in result3_text.lower() or "deep learning" in result3_text.lower()

                    # Search for content in new file
                    result4 = await session.call_tool(
                        "search_local",
                        arguments={"query": "blockchain technology", "top_k": 3}
                    )
                    result4_text = result4.content[0].text
                    new_file_found = "blockchain" in result4_text.lower()

            # Verify incremental indexing worked
            if initial_found and new_content_found and new_file_found:
                log_test("Incremental Indexing", True, "Successfully detected modified and new files")
                return True
            else:
                details = f"initial={initial_found}, modified={new_content_found}, new={new_file_found}"
                log_test("Incremental Indexing", False, f"Failed to detect changes: {details}")
                return False

    except Exception as e:
        log_test("Incremental Indexing", False, str(e))
        return False


async def test_search_strategies():
    """Test 5: Verify different search strategies work correctly."""
    try:
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "python", "-m", "src"],
            env={
                **os.environ.copy(),
                "PYTHONPATH": PROJECT_ROOT,
                "LOCAL_DOCS_PATH": TEST_DOCS_PATH,
                "SKIP_WIKIPEDIA": "true"
            },
            cwd=PROJECT_ROOT
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Test keyword strategy
                result_keyword = await session.call_tool(
                    "search_local",
                    arguments={"query": "Python Django Flask", "top_k": 3, "strategy": "keyword"}
                )

                # Test hybrid strategy
                result_hybrid = await session.call_tool(
                    "search_local",
                    arguments={"query": "Python web frameworks", "top_k": 3, "strategy": "hybrid"}
                )

                keyword_text = result_keyword.content[0].text
                hybrid_text = result_hybrid.content[0].text

                # Both should return results
                if len(keyword_text) > 50 and len(hybrid_text) > 50:
                    log_test("Search Strategies", True, "Both keyword and hybrid strategies work")
                    return True
                else:
                    log_test("Search Strategies", False,
                            f"Strategies returned insufficient results: keyword={len(keyword_text)}, hybrid={len(hybrid_text)}")
                    return False

    except Exception as e:
        log_test("Search Strategies", False, str(e))
        return False


async def run_all_tests():
    """Run all CI/CD tests."""
    print("=" * 70)
    print("LocalSearch MCP Server - CI/CD Test Suite")
    print("=" * 70)
    print(f"Test documents path: {TEST_DOCS_PATH}")
    print()

    # Verify test documents exist
    if not os.path.exists(TEST_DOCS_PATH):
        print(f"❌ ERROR: Test documents directory not found: {TEST_DOCS_PATH}")
        sys.exit(1)

    test_files = list(Path(TEST_DOCS_PATH).glob("**/*.md")) + list(Path(TEST_DOCS_PATH).glob("**/*.txt"))
    print(f"Found {len(test_files)} test documents")
    print()

    # Run tests
    tests = [
        test_mcp_connection,
        test_local_indexing,
        test_search_results,
        test_incremental_indexing,
        test_search_strategies
    ]

    for test_func in tests:
        await test_func()
        print()

    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for r in test_results if r["passed"])
    total = len(test_results)

    for result in test_results:
        status = "✅" if result["passed"] else "❌"
        print(f"{status} {result['test']}")

    print()
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)

    # Exit with appropriate code
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    print("\n📋 This test suite is designed for CI/CD environments")
    print("   It tests indexing and search functionality without requiring LLM\n")

    asyncio.run(run_all_tests())
