# Local Search MCP Server

A standalone, offline Wikipedia search server implementing the Model Context Protocol (MCP). This server enables AI assistants to search through locally-indexed Wikipedia content without requiring external API calls or internet connectivity.

[日本語版 README はこちら](#日本語版)

## Features

- **Completely Offline**: No external API dependencies (Google Search, etc.)
- **Free & Fast**: Uses BM25 algorithm for efficient full-text search
- **MCP Compatible**: Works with any MCP-compatible client (Claude Desktop, etc.)
- **Ollama Integration**: Includes test client for Ollama-based agents
- **Easy Setup**: Simple installation with `uv` package manager

## Architecture

```
┌─────────────┐         ┌──────────────────┐         ┌─────────────┐
│   Ollama    │ ◄────── │  MCP Client      │ ◄────── │   Human     │
│   (LLM)     │         │  (test script)   │         │             │
└─────────────┘         └──────────────────┘         └─────────────┘
                                │
                                │ MCP Protocol
                                ▼
                        ┌──────────────────┐
                        │  MCP Server      │
                        │  (src/server.py) │
                        └──────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │  BM25 Index      │
                        │  (Wikipedia)     │
                        └──────────────────┘
```

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- (Optional) Ollama with a tool-compatible model (e.g., command-r) for testing

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/localsearch-mcp.git
cd localsearch-mcp
```

2. Install dependencies:
```bash
uv sync
```

3. Build the Wikipedia index (first run only):
```bash
uv run python -m src
# Press Ctrl+C after index is built
```

This will download English Wikipedia (~6.8M articles, ~20GB) and create a BM25 index in the `data/` directory. The initial build takes significant time and disk space.

## Usage

### Running the MCP Server

```bash
uv run python -m src
```

The server will:
1. Load the pre-built Wikipedia index
2. Start listening for MCP requests on stdio
3. Provide the `search_wikipedia` tool

### Testing with Ollama

#### Simple Test (No LLM)
```bash
uv run tests/verify_with_ollama.py --simple
```

This tests the MCP connection and performs a direct search.

#### Full Agent Test (Requires Ollama)
```bash
# Make sure Ollama is running with a tool-compatible model
ollama pull command-r
ollama serve

# In another terminal:
uv run tests/verify_with_ollama.py
```

Expected output:
```
🤖 Starting MCP Client and connecting to Local Search Server...
✅ Connected. Available tools: ['search_wikipedia']

👤 User Query: Pythonというプログラミング言語の歴史について、簡潔に教えて
🛠️  Agent requested 1 tool call(s)
   → Tool: search_wikipedia
   → Args: {'query': 'history of python programming language'}
   → Output length: 1523 chars

🤖 Agent Answer:
Python was created by Guido van Rossum in the late 1980s...
```

### Integration with Claude Desktop

Add this to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "local-wiki-search": {
      "command": "uv",
      "args": ["run", "python", "-m", "src"],
      "cwd": "/path/to/localsearch-mcp"
    }
  }
}
```

Then restart Claude Desktop and you can use the Wikipedia search tool in your conversations.

## Project Structure

```
localsearch-mcp/
├── pyproject.toml          # Dependencies and project metadata
├── README.md               # This file
├── data/                   # Index storage (created on first run)
│   ├── .gitkeep
│   └── wiki_index.pkl      # BM25 index (not in git)
├── src/
│   ├── __init__.py
│   ├── __main__.py         # Entry point for `python -m src`
│   ├── server.py           # MCP server implementation
│   └── indexer.py          # BM25 indexing logic
└── tests/
    ├── __init__.py
    └── verify_with_ollama.py  # Ollama integration test client
```

## Available Tools

### `search_wikipedia`

Search English Wikipedia for a given query using BM25 algorithm.

**Parameters:**
- `query` (string, required): Search keywords
- `top_k` (integer, optional): Number of results to return (default: 3, max: 10)

**Returns:**
Formatted search results with titles, Wikipedia URLs, and content snippets.

**Example:**
```python
# MCP tool call
result = await session.call_tool(
    "search_wikipedia",
    arguments={"query": "python programming language", "top_k": 3}
)
```

## Customization

### Using Simple English Wikipedia (for development)

For faster development/testing, use the lightweight Simple English Wikipedia:

Edit `src/indexer.py`:
```python
# Change this line:
ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

# To (Simple English, limited to 10k articles):
ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train[:10000]")
```

This reduces disk space to ~500MB and builds in a few minutes.

### Adjusting Index Size

You can limit the number of articles for testing:

```python
# Limit to 1000 articles
ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:1000]")
```

## Development

### Running Tests
```bash
# Simple MCP connection test
uv run tests/verify_with_ollama.py --simple

# Full Ollama agent test
uv run tests/verify_with_ollama.py
```

### Rebuilding Index
Delete `data/wiki_index.pkl` and restart the server.

## Troubleshooting

### Index Not Building
- Check disk space (needs ~500MB for Simple Wikipedia, ~20GB for full)
- Ensure stable internet connection for initial download
- Check Python version (3.10+ required)

### Ollama Connection Fails
- Verify Ollama is running: `ollama list`
- Ensure a tool-compatible model is installed: `ollama pull command-r`
- Check Ollama API is accessible: `curl http://localhost:11434`

### MCP Server Not Starting
- Check dependencies: `uv sync`
- Verify Python path in MCP config
- Check for port conflicts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

---

# 日本語版

## 概要

ローカル環境で動作する Wikipedia 検索 MCP サーバーです。外部 API に依存せず、完全にオフラインで動作します。

## 特徴

- **完全オフライン**: インターネット接続不要
- **無料・高速**: BM25 アルゴリズムによる効率的な全文検索
- **MCP 互換**: Claude Desktop などの MCP 対応クライアントで使用可能
- **Ollama 統合**: Ollama を使ったテストクライアント付属
- **簡単セットアップ**: `uv` による簡単インストール

## インストール

### 必要要件

- Python 3.10 以上
- [uv](https://github.com/astral-sh/uv) パッケージマネージャー
- (オプション) テスト用の Ollama とツール対応モデル（例: command-r）

### セットアップ手順

1. リポジトリをクローン:
```bash
git clone https://github.com/yourusername/localsearch-mcp.git
cd localsearch-mcp
```

2. 依存関係をインストール:
```bash
uv sync
```

3. Wikipedia インデックスを構築（初回のみ）:
```bash
uv run python -m src
# インデックス構築後 Ctrl+C で終了
```

これにより英語版 Wikipedia（約680万記事、約20GB）がダウンロードされ、`data/` ディレクトリに BM25 インデックスが作成されます。初回構築には時間とディスク容量が必要です。

## 使い方

### MCP サーバーの起動

```bash
uv run src/server.py
```

サーバーは以下を実行します：
1. 構築済み Wikipedia インデックスを読み込み
2. 標準入出力で MCP リクエストを待機
3. `search_wikipedia` ツールを提供

### Ollama を使ったテスト

#### シンプルテスト（LLM なし）
```bash
uv run tests/verify_with_ollama.py --simple
```

MCP 接続と検索機能をテストします。

#### エージェントテスト（Ollama 必要）
```bash
# Ollama と llama3.2 モデルを起動
ollama pull llama3.2
ollama serve

# 別のターミナルで実行:
uv run tests/verify_with_ollama.py
```

### Claude Desktop との統合

Claude Desktop の MCP 設定に以下を追加:

```json
{
  "mcpServers": {
    "local-wiki-search": {
      "command": "uv",
      "args": ["run", "/絶対パス/localsearch-mcp/src/server.py"]
    }
  }
}
```

Claude Desktop を再起動すると、会話内で Wikipedia 検索が使えるようになります。

## 利用可能なツール

### `search_wikipedia`

BM25 アルゴリズムを使って Wikipedia を検索します。

**パラメータ:**
- `query` (文字列, 必須): 検索キーワード
- `top_k` (整数, オプション): 返す結果の数（デフォルト: 3、最大: 10）

**戻り値:**
タイトル、URL、本文スニペットを含む検索結果

## カスタマイズ

### 完全版 Wikipedia を使用

`src/indexer.py` を編集:
```python
# この行を変更:
ds = load_dataset("wikipedia", "20220301.simple", split="train[:10000]")

# 以下に変更:
ds = load_dataset("wikipedia", "20231101.en", split="train")
```

注: 約20GB のディスクスペースと長い構築時間が必要です。

## トラブルシューティング

### インデックスが構築されない
- ディスク容量を確認（Simple 版で約500MB、完全版で約20GB必要）
- 初回ダウンロード用のインターネット接続を確認
- Python バージョンを確認（3.10以上必要）

### Ollama 接続エラー
- Ollama が起動しているか確認: `ollama list`
- llama3.2 がインストールされているか確認: `ollama pull llama3.2`
- Ollama API にアクセス可能か確認: `curl http://localhost:11434`

## ライセンス

MIT License
