"""
Ollama Integration Test for LocalKB MCP Server

⚠️  LOCAL TESTING ONLY - NOT FOR CI/CD ⚠️

This script acts as an MCP client that uses Ollama for LLM inference.
It demonstrates the complete workflow: query -> tool use -> final answer.

Requirements:
- Ollama must be installed and running locally
- Required models: llama3.2, command-r (or compatible models)
- Test documents: Uses test_docs/ directory by default (Python, ML, Database, Web Dev content)
  - Can be overridden with LOCAL_DOCS_PATH environment variable

This test is intended for local development and validation only.
For CI/CD testing, use test_indexing_search.py instead.
"""
import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama

# MCP Server module path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Local documents path for testing
# Default: Use test_docs/ (repository test documents with Python, ML, Database, Web Dev content)
# Override with LOCAL_DOCS_PATH environment variable if needed
TEST_DOCS_DIR = os.path.join(PROJECT_ROOT, "test_docs")
LOCAL_DOCS_PATH = os.getenv("LOCAL_DOCS_PATH", TEST_DOCS_DIR)


async def run_ollama_agent():
    """
    Main agent loop that:
    1. Connects to Local Search MCP Server
    2. Sends query to Ollama
    3. Executes tool calls via MCP
    4. Returns final answer
    """
    # Configure MCP server parameters
    server_params = StdioServerParameters(
        command="uv",  # or "python" if not using uv
        args=["run", "python", "-m", "src"],
        env={**os.environ.copy(), "PYTHONPATH": PROJECT_ROOT, "LOCAL_DOCS_PATH": LOCAL_DOCS_PATH},
        cwd=PROJECT_ROOT
    )

    print("🤖 Starting MCP Client and connecting to Local Search Server...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize MCP session
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print(f"✅ Connected. Available tools: {[t.name for t in tools.tools]}")

            # Convert MCP tools to Ollama function calling format
            ollama_tools = []
            for tool in tools.tools:
                ollama_tools.append({
                    'type': 'function',
                    'function': {
                        'name': tool.name,
                        'description': tool.description,
                        'parameters': tool.inputSchema
                    }
                })

            # User query (using test_docs content)
            query = "What are the popular Python web frameworks?"
            print(f"\n👤 User Query: {query}")

            messages = [{'role': 'user', 'content': query}]

            # First Ollama call - agent decides whether to use tools
            print("\n🔄 Calling Ollama...")
            response = ollama.chat(
                model='llama3.2',  # Use function calling compatible model
                messages=messages,
                tools=ollama_tools
            )

            # Check if agent wants to use tools
            if response.message.tool_calls:
                print(f"\n🛠️  Agent requested {len(response.message.tool_calls)} tool call(s)")

                # Execute each tool call via MCP
                for tool_call in response.message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = tool_call.function.arguments
                    print(f"   → Tool: {fn_name}")
                    print(f"   → Args: {fn_args}")

                    # Call MCP tool
                    result = await session.call_tool(fn_name, arguments=fn_args)
                    tool_output = result.content[0].text
                    print(f"   → Output length: {len(tool_output)} chars")
                    print(f"   → Output: {tool_output}...")

                    # Append tool results to message history
                    messages.append(response.message)
                    messages.append({
                        'role': 'tool',
                        'content': tool_output,
                        'name': fn_name
                    })

                # Second Ollama call - synthesize final answer using tool results
                print("\n🔄 Generating final answer...")
                final_response = ollama.chat(
                    model='command-r',
                    messages=messages
                )
                print(f"\n🤖 Agent Answer:\n{final_response.message.content}")
            else:
                # Agent answered directly without tools
                print("\n⚠️  Agent did not use any tools.")
                print(f"\n🤖 Agent Answer:\n{response.message.content}")


async def run_simple_test():
    """Simple test that directly calls the search tool without LLM."""
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "src"],
        env={**os.environ.copy(), "PYTHONPATH": PROJECT_ROOT, "LOCAL_DOCS_PATH": LOCAL_DOCS_PATH},
        cwd=PROJECT_ROOT
    )

    print("🧪 Running simple MCP connection test...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            print(f"✅ Available tools: {[t.name for t in tools.tools]}")

            # Wikipedia search test
            print("\n🔍 Testing Wikipedia search: 'Python programming language'")
            result = await session.call_tool(
                "search_offline_wikipedia",
                arguments={"query": "Python programming language", "top_k": 2}
            )
            print(f"\n📄 Wikipedia Result:\n{result.content[0].text}")


async def run_local_search_test():
    """Test local document search with test_docs documents."""
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "src"],
        env={**os.environ.copy(), "PYTHONPATH": PROJECT_ROOT, "LOCAL_DOCS_PATH": LOCAL_DOCS_PATH},
        cwd=PROJECT_ROOT
    )

    print("🧪 Running Local Document Search Test (test_docs)...")
    print(f"📁 Local docs path: {LOCAL_DOCS_PATH}\n")

    # Test queries that can be answered from test_docs
    test_queries = [
        {
            "query": "Python web frameworks Django Flask",
            "expected_keywords": ["Django", "Flask"],
            "description": "Python web frameworks"
        },
        {
            "query": "machine learning algorithms supervised unsupervised",
            "expected_keywords": ["supervised", "unsupervised", "classification", "regression"],
            "description": "Machine learning types"
        },
        {
            "query": "database management system DBMS SQL",
            "expected_keywords": ["database", "DBMS", "SQL"],
            "description": "Database management systems"
        },
        {
            "query": "web development frontend backend HTML CSS JavaScript",
            "expected_keywords": ["frontend", "backend", "HTML", "CSS", "JavaScript"],
            "description": "Web development technologies"
        },
    ]

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"✅ Available tools: {[t.name for t in tools.tools]}")

            if "search_internal_technical_documents" not in [t.name for t in tools.tools]:
                print("\n⚠️  Warning: search_internal_technical_documents tool not available!")
                print("   Make sure LOCAL_DOCS_PATH is set correctly.")
                return

            print("\n" + "="*60)
            print("📚 Testing Local Document Search Queries")
            print("="*60)

            passed = 0
            for i, test in enumerate(test_queries, 1):
                print(f"\n--- Test {i}: {test['description']} ---")
                print(f"🔍 Query: {test['query']}")
                print(f"📋 Expected keywords: {', '.join(test['expected_keywords'])}")

                result = await session.call_tool(
                    "search_internal_technical_documents",
                    arguments={"query": test["query"], "top_k": 3}
                )

                result_text = result.content[0].text
                print(f"\n📄 Search Result (first 500 chars):")
                print(result_text[:500] if len(result_text) > 500 else result_text)

                # Check if at least one expected keyword is in results
                found_keywords = [kw for kw in test["expected_keywords"] if kw.lower() in result_text.lower()]
                if found_keywords:
                    print(f"\n✅ PASS: Found keywords: {', '.join(found_keywords)}")
                    passed += 1
                else:
                    print(f"\n⚠️  FAIL: No expected keywords found")

                print("-"*40)

            print(f"\n📊 Results: {passed}/{len(test_queries)} queries passed")

            print("\n" + "="*60)
            print("🔍 Testing Multi-Source Search (Wikipedia + Local)")
            print("="*60)

            # Test combined search (should return both Wikipedia and local results)
            result = await session.call_tool(
                "query_internal_knowledge_base",
                arguments={"query": "Python programming language web frameworks", "top_k": 3, "source": "all"}
            )
            result_text = result.content[0].text
            print(f"\n📄 Combined Search Result (first 1000 chars):")
            print(result_text[:1000] if len(result_text) > 1000 else result_text)
            print("\n(Should contain both Wikipedia results and local test_docs results)")


async def run_local_qa_test():
    """
    Test local file Q&A with Ollama.
    Uses local document search (test_docs) and Ollama to generate answers based on local files only.
    Tests the complete RAG (Retrieval-Augmented Generation) workflow.
    """
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "src"],
        env={**os.environ.copy(), "PYTHONPATH": PROJECT_ROOT, "LOCAL_DOCS_PATH": LOCAL_DOCS_PATH},
        cwd=PROJECT_ROOT
    )

    print("🧪 Running Local File Q&A Test with Ollama (RAG Demo)...")
    print(f"📁 Local docs path: {LOCAL_DOCS_PATH}")
    print("📚 Test documents contain info about Python, ML, Databases, and Web Dev\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"✅ Connected. Available tools: {[t.name for t in tools.tools]}\n")

            if "search_internal_technical_documents" not in [t.name for t in tools.tools]:
                print("\n⚠️  Warning: search_internal_technical_documents tool not available!")
                return

            # Test questions based on test_docs content
            test_cases = [
                {
                    "question": "What are the popular Python web frameworks mentioned in the documents?",
                    "search_query": "Python web frameworks Django Flask"
                },
                {
                    "question": "What are the main types of machine learning?",
                    "search_query": "machine learning types supervised unsupervised reinforcement"
                },
                {
                    "question": "What technologies are used in web development?",
                    "search_query": "web development frontend backend technologies"
                }
            ]

            # Select a test case
            test = test_cases[0]
            question = test["question"]
            search_query = test["search_query"]

            print(f"👤 Question: {question}")
            print(f"🔍 Search Query: {search_query}\n")

            # Step 1: Search local files with optimized keyword query
            print(f"🔍 Step 1: Searching local files...")
            result = await session.call_tool(
                "search_internal_technical_documents",
                arguments={"query": search_query, "top_k": 3, "strategy": "keyword"}
            )

            search_results_full = result.content[0].text

            # Truncate to first 8000 chars to avoid overwhelming the model
            max_chars = 8000
            search_results = search_results_full[:max_chars]
            if len(search_results_full) > max_chars:
                search_results += "\n\n[... results truncated ...]"

            print(f"   Found {len(search_results_full)} chars, using {len(search_results)} chars")

            # Debug: Check if expected keywords are in results
            expected_keywords = ["Django", "Flask", "Python"]
            found = [kw for kw in expected_keywords if kw in search_results]
            if found:
                print(f"   ✓ Search results contain: {', '.join(found)}\n")
            else:
                print("   ✗ Search results may not contain expected keywords\n")

            # Step 2: Use Ollama to generate answer based on search results
            print("🤖 Step 2: Generating answer with Ollama...")

            # Create prompt with search results
            prompt = f"""You are analyzing technical documentation. Answer the question based ONLY on the search results provided.

Question: {question}

Search Results:
{search_results}

Instructions:
- Use only information from the search results
- If you find specific numbers or specifications, state them clearly
- Keep your answer concise (1-2 sentences)
- If the answer is not found, say "Information not found"

Answer:"""

            # Call Ollama
            response = ollama.chat(
                model='llama3.2',  # Use llama3.2 for Q&A
                messages=[{'role': 'user', 'content': prompt}]
            )

            answer = response.message.content

            print("\n" + "="*60)
            print("📝 Final Answer:")
            print("="*60)
            print(answer)
            print("="*60)

            # Show source information
            print(f"\n💡 Answer generated from {LOCAL_DOCS_PATH}")


if __name__ == "__main__":
    print("=" * 60)
    print("LocalKB MCP Server - Ollama Integration Test")
    print("=" * 60)

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--simple":
            # Run simple test without Ollama
            asyncio.run(run_simple_test())
        elif sys.argv[1] == "--local":
            # Run local document search test
            asyncio.run(run_local_search_test())
        elif sys.argv[1] == "--local-qa":
            # Run local file Q&A with Ollama
            print("\nℹ️  This requires Ollama to be running with llama3.2 model")
            print("   Install with: ollama pull llama3.2")
            print(f"   Using test documents from: {LOCAL_DOCS_PATH}\n")
            asyncio.run(run_local_qa_test())
        else:
            print(f"\n❌ Unknown option: {sys.argv[1]}")
            print("   Available options:")
            print("   --simple    : Test MCP connection with Wikipedia search")
            print("   --local     : Test local document search (test_docs)")
            print("   --local-qa  : Test local file Q&A with Ollama (uses llama3.2)")
            print("   (no args)   : Full Ollama agent test with function calling")
    else:
        # Run full Ollama agent test
        print("\nℹ️  This requires Ollama to be running with llama3.2 model")
        print("   Install with: ollama pull llama3.2 command-r")
        print("   Use --simple flag to test MCP connection only")
        print("   Use --local flag to test local document search")
        print("   Use --local-qa flag to test Q&A with local files")
        print(f"   Using test documents from: {LOCAL_DOCS_PATH}\n")
        asyncio.run(run_ollama_agent())
