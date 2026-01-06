"""
Ollama Integration Test for Local Search MCP Server

This script acts as an MCP client that uses Ollama for LLM inference.
It demonstrates the complete workflow: query -> tool use -> final answer.
"""
import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama

# MCP Server module path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Local documents path for VisionSort/Casper KB
LOCAL_DOCS_PATH = "/Users/ikmx/source/tc/Casper_KB-main/docs/consolidated"


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

            # User query
            query = "VisionSort 405nm laser output power mW"
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
                "search_wikipedia",
                arguments={"query": "Python programming language", "top_k": 2}
            )
            print(f"\n📄 Wikipedia Result:\n{result.content[0].text}")


async def run_local_search_test():
    """Test local document search with VisionSort/Casper KB documents."""
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "src"],
        env={**os.environ.copy(), "PYTHONPATH": PROJECT_ROOT, "LOCAL_DOCS_PATH": LOCAL_DOCS_PATH},
        cwd=PROJECT_ROOT
    )

    print("🧪 Running Local Document Search Test (VisionSort/Casper KB)...")
    print(f"📁 Local docs path: {LOCAL_DOCS_PATH}\n")

    # Test queries that can ONLY be answered from local documents
    test_queries = [
        {
            "query": "VisionSort 405nm laser output power mW",
            "expected_answer": "365 mW",
            "description": "VisionSort 405nmレーザーの出力"
        },
        {
            "query": "FluidicSystem error code 4015 CL Leak",
            "expected_answer": "Emergency level, chip holder leak",
            "description": "エラーコード4015の意味と対処法"
        },
        {
            "query": "Ultrafine focus step size micrometer",
            "expected_answer": "0.05μm",
            "description": "Ultrafineフォーカスのステップサイズ"
        },
        {
            "query": "VisionSort sorting cartridge channel dimension",
            "expected_answer": "34 × 50 µm",
            "description": "ソーティングカートリッジのチャンネル寸法"
        },
        {
            "query": "recommended Edgemode focus calculation ComprehensiveQuality",
            "expected_answer": "ComprehensiveQuality",
            "description": "推奨されるフォーカス計算のEdgemode"
        },
    ]

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"✅ Available tools: {[t.name for t in tools.tools]}")

            if "search_local" not in [t.name for t in tools.tools]:
                print("\n⚠️  Warning: search_local tool not available!")
                print("   Make sure LOCAL_DOCS_PATH is set correctly.")
                return

            print("\n" + "="*60)
            print("📚 Testing Local Document Search Queries")
            print("="*60)

            for i, test in enumerate(test_queries, 1):
                print(f"\n--- Test {i}: {test['description']} ---")
                print(f"🔍 Query: {test['query']}")
                print(f"📋 Expected: {test['expected_answer']}")

                result = await session.call_tool(
                    "search_local",
                    arguments={"query": test["query"], "top_k": 3}
                )

                result_text = result.content[0].text
                print(f"\n📄 Search Result (first 500 chars):")
                print(result_text[:500] if len(result_text) > 500 else result_text)

                # Check if expected answer is in results
                if test["expected_answer"].lower() in result_text.lower():
                    print(f"\n✅ PASS: Expected answer found in results!")
                else:
                    print(f"\n⚠️  CHECK: Expected answer may not be directly visible")

                print("-"*40)

            print("\n" + "="*60)
            print("🔍 Testing Multi-Source Search (Wikipedia + Local)")
            print("="*60)

            # Test combined search
            result = await session.call_tool(
                "search",
                arguments={"query": "cell sorting flow cytometry", "top_k": 3, "source": "all"}
            )
            print(f"\n📄 Combined Search Result:\n{result.content[0].text[:1000]}...")


async def run_local_qa_test():
    """
    Test local file Q&A with Ollama.
    Uses local document search and Ollama to generate answers based on local files only.
    """
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "src"],
        env={**os.environ.copy(), "PYTHONPATH": PROJECT_ROOT, "LOCAL_DOCS_PATH": LOCAL_DOCS_PATH},
        cwd=PROJECT_ROOT
    )

    print("🧪 Running Local File Q&A Test with Ollama...")
    print(f"📁 Local docs path: {LOCAL_DOCS_PATH}\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"✅ Connected. Available tools: {[t.name for t in tools.tools]}\n")

            if "search_local" not in [t.name for t in tools.tools]:
                print("\n⚠️  Warning: search_local tool not available!")
                return

            # Test questions in English with optimized search queries
            test_cases = [
                {
                    "question": "What is the output power in mW of the 405nm laser used in VisionSort?",
                    "search_query": "405nm laser output power mW VisionSort"
                },
                {
                    "question": "What does FluidicSystem error code 4015 mean and how to fix it?",
                    "search_query": "FluidicSystem error code 4015"
                },
                {
                    "question": "What are the channel dimensions of the VisionSort sorting cartridge?",
                    "search_query": "VisionSort sorting cartridge channel dimensions"
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
                "search_local",
                arguments={"query": search_query, "top_k": 3, "strategy": "keyword"}
            )

            search_results_full = result.content[0].text

            # Truncate to first 8000 chars to avoid overwhelming the model
            max_chars = 8000
            search_results = search_results_full[:max_chars]
            if len(search_results_full) > max_chars:
                search_results += "\n\n[... results truncated ...]"

            print(f"   Found {len(search_results_full)} chars, using {len(search_results)} chars")

            # Debug: Check if answer is in results
            if "365" in search_results:
                print("   ✓ Search results contain '365'\n")
            else:
                print("   ✗ Search results do NOT contain '365'\n")

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
    print("Local Search MCP Server - Ollama Integration Test")
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
            print("   Install with: ollama pull llama3.2\n")
            asyncio.run(run_local_qa_test())
        else:
            print(f"\n❌ Unknown option: {sys.argv[1]}")
            print("   Available options:")
            print("   --simple    : Test MCP connection with Wikipedia search")
            print("   --local     : Test local document search (VisionSort/Casper KB)")
            print("   --local-qa  : Test local file Q&A with Ollama (uses llama3.2)")
            print("   (no args)   : Full Ollama agent test with function calling")
    else:
        # Run full Ollama agent test
        print("\nℹ️  This requires Ollama to be running with llama3.2 model")
        print("   Use --simple flag to test MCP connection only")
        print("   Use --local flag to test local document search")
        print("   Use --local-qa flag to test Q&A with local files\n")
        asyncio.run(run_ollama_agent())
