# Multi-Source Local Search MCP Server

A standalone, offline search server implementing the Model Context Protocol (MCP). This server enables AI assistants to search through **Wikipedia (static, large-scale knowledge)** and **your local files (dynamic, personal knowledge)** without requiring external API calls or internet connectivity.

[日本語版 README はこちら](#日本語版)

## Features

- **Multi-Source Search**: Search across Wikipedia AND your local files (Markdown, text) simultaneously
- **Hybrid Search**: Combines BM25 (keyword matching) + Vector embeddings (semantic similarity) for best results
- **Smart Indexing**: Wikipedia index cached permanently, local files scanned on startup for latest changes
- **Completely Offline**: No external API dependencies (Google Search, etc.)
- **Free & Fast**: Uses efficient algorithms for both keyword and semantic search
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
                   ┌────────────┴────────────┐
                   ▼                         ▼
         ┌──────────────────┐      ┌──────────────────┐
         │ Wikipedia Indexer│      │ Local File       │
         │ (Static/Cached)  │      │ Indexer (Dynamic)│
         └──────────────────┘      └──────────────────┘
                   │                         │
                   ▼                         ▼
         ┌──────────────────┐      ┌──────────────────┐
         │ BM25 + Vector DB │      │ BM25 + Vector DB │
         │ (1M+ articles)   │      │ (Your files)     │
         └──────────────────┘      └──────────────────┘
```

**Composite Pattern**: Results from both sources are merged using Reciprocal Rank Fusion (RRF) for optimal ranking.

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
# Set smaller subset for testing (optional)
export WIKI_SUBSET_SIZE=10000  # Default: 1,000,000

uv run python -m src
# Press Ctrl+C after index is built
```

This will download English Wikipedia and create:
- **BM25 index** (keyword search) in `data/wiki_index.pkl`
- **Vector index** (semantic search) in `data/chroma_db/`

The initial build downloads documents and generates embeddings, which takes time. Default: 1M articles (~5GB). Full dataset: 6.8M articles (~20GB).

4. (Optional) Enable local file search:
```bash
# Set the path to your local documents
export LOCAL_DOCS_PATH="/path/to/your/notes"  # e.g., ~/ObsidianVault/Research
```

This enables searching through your:
- Markdown files (`.md`)
- Text files (`.txt`)
- Any personal notes or documentation

The server will scan this directory on each startup to index the latest content.

## Usage

### Running the MCP Server

```bash
# Without local files
uv run python -m src

# With local files
LOCAL_DOCS_PATH="/path/to/your/notes" uv run python -m src
```

The server will:
1. Load the pre-built Wikipedia index (cached, fast)
2. Scan and index local files if `LOCAL_DOCS_PATH` is set (quick for typical document collections)
3. Start listening for MCP requests on stdio
4. Provide search tools: `search`, `search_wikipedia`, and `search_local`

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

**Wikipedia only:**
```json
{
  "mcpServers": {
    "local-search": {
      "command": "uv",
      "args": ["run", "python", "-m", "src"],
      "cwd": "/path/to/localsearch-mcp"
    }
  }
}
```

**Wikipedia + Local Files:**
```json
{
  "mcpServers": {
    "local-search": {
      "command": "uv",
      "args": ["run", "python", "-m", "src"],
      "cwd": "/path/to/localsearch-mcp",
      "env": {
        "LOCAL_DOCS_PATH": "/Users/yourname/Documents/Notes"
      }
    }
  }
}
```

Then restart Claude Desktop and you can search both Wikipedia and your personal files in conversations!

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

### `search` (Multi-Source)

Search across Wikipedia AND your local files simultaneously using hybrid search.

**Parameters:**
- `query` (string, required): Search keywords or question
- `top_k` (integer, optional): Number of results to return per source (default: 5, max: 20)
- `strategy` (string, optional): Search strategy - `"hybrid"` (default), `"keyword"`, or `"semantic"`
- `source` (string, optional): Data source - `"all"` (default), `"wikipedia"`, or `"local"`

**Source Options:**
- **`"all"`** (default): Search both Wikipedia and local files for comprehensive results
- **`"wikipedia"`**: Search only Wikipedia (general knowledge)
- **`"local"`**: Search only your local files (personal knowledge)

**Search Strategies:**
- **`"hybrid"`** (recommended): Combines keyword matching and semantic similarity for best results
- **`"keyword"`**: Traditional BM25 keyword search (exact word matching, fast)
- **`"semantic"`**: Vector similarity search (finds conceptually similar content, even without exact words)

**Returns:**
Formatted search results with titles, URLs/paths, and content snippets. Results from both sources are merged intelligently using Reciprocal Rank Fusion (RRF).

### `search_wikipedia`

Search English Wikipedia only using hybrid search (BM25 + Vector embeddings). Convenience wrapper for `search` with `source="wikipedia"`.

**Parameters:**
- `query` (string, required): Search keywords or question
- `top_k` (integer, optional): Number of results to return (default: 3, max: 10)
- `strategy` (string, optional): Search strategy - `"hybrid"` (default), `"keyword"`, or `"semantic"`

### `search_local`

Search your local files only using hybrid search. Convenience wrapper for `search` with `source="local"`.

**Parameters:**
- `query` (string, required): Search keywords or question
- `top_k` (integer, optional): Number of results to return (default: 5, max: 20)
- `strategy` (string, optional): Search strategy - `"hybrid"` (default), `"keyword"`, or `"semantic"`

**Examples:**
```python
# Hybrid search (best results, default)
result = await session.call_tool(
    "search_wikipedia",
    arguments={"query": "python programming language", "top_k": 3}
)

# Keyword-only search (fast, exact matches)
result = await session.call_tool(
    "search_wikipedia",
    arguments={"query": "python programming language", "strategy": "keyword"}
)

# Semantic search (finds similar concepts)
result = await session.call_tool(
    "search_wikipedia",
    arguments={"query": "snake that inspired a programming language", "strategy": "semantic"}
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

- **ハイブリッド検索**: BM25（キーワード検索）+ ベクトル埋め込み（意味検索）の組み合わせで最高の結果を提供
- **完全オフライン**: インターネット接続不要
- **無料・高速**: キーワードと意味の両方に対応した効率的な検索アルゴリズム
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
# テスト用に小さいサブセットを使用（オプション）
export WIKI_SUBSET_SIZE=10000  # デフォルト: 1,000,000

uv run python -m src
# インデックス構築後 Ctrl+C で終了
```

これにより以下が作成されます：
- **BM25 インデックス**（キーワード検索）: `data/wiki_index.pkl`
- **ベクトルインデックス**（意味検索）: `data/chroma_db/`

初回構築はドキュメントのダウンロードと埋め込み生成を行うため時間がかかります。デフォルト: 100万記事（約5GB）。完全版: 680万記事（約20GB）。

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

ハイブリッド検索（BM25 + ベクトル埋め込み）を使って Wikipedia を検索します。

**パラメータ:**
- `query` (文字列, 必須): 検索キーワードまたは質問
- `top_k` (整数, オプション): 返す結果の数（デフォルト: 3、最大: 10）
- `strategy` (文字列, オプション): 検索戦略 - `"hybrid"` (デフォルト)、`"keyword"`、または `"semantic"`

**検索戦略:**
- **`"hybrid"`** (推奨): キーワード検索と意味検索を組み合わせて最良の結果を提供
- **`"keyword"`**: 従来の BM25 キーワード検索（完全一致、高速）
- **`"semantic"`**: ベクトル類似度検索（単語が一致しなくても概念的に類似したコンテンツを検索）

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
