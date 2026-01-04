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

# MCP Server script path
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "../src/server.py")


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
        args=["run", SERVER_SCRIPT],
        env=os.environ.copy()
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
            query = "Pythonというプログラミング言語の歴史について、簡潔に教えて"
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
                    model='llama3.2',
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
        args=["run", SERVER_SCRIPT],
        env=os.environ.copy()
    )

    print("🧪 Running simple MCP connection test...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            print(f"✅ Available tools: {[t.name for t in tools.tools]}")

            # Direct tool call
            print("\n🔍 Testing direct search: 'Python programming language'")
            result = await session.call_tool(
                "search_wikipedia",
                arguments={"query": "Python programming language", "top_k": 2}
            )
            print(f"\n📄 Search Result:\n{result.content[0].text}")


if __name__ == "__main__":
    print("=" * 60)
    print("Local Search MCP Server - Ollama Integration Test")
    print("=" * 60)

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        # Run simple test without Ollama
        asyncio.run(run_simple_test())
    else:
        # Run full Ollama agent test
        print("\nℹ️  This requires Ollama to be running with llama3.2 model")
        print("   Use --simple flag to test MCP connection only\n")
        asyncio.run(run_ollama_agent())
