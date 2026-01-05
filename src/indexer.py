"""
Wikipedia Hybrid Search Indexer for Local Search MCP Server
Handles downloading, tokenizing, and indexing Wikipedia data.
Supports both BM25 (keyword) and vector (semantic) search.
"""
import os
import pickle
import sys
from typing import List, Dict, Literal
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

INDEX_PATH = "data/wiki_index.pkl"
CHROMA_PATH = "data/chroma_db"

# Number of documents to index. Set via WIKI_SUBSET_SIZE env var.
# Default: 1,000,000 (sufficient for HotPotQA evaluation)
# Full English Wikipedia: ~6.4M articles
# For development/testing: 10,000-50,000
DEFAULT_SUBSET_SIZE = 1_000_000

class WikiIndexer:
    """Builds and manages hybrid search index (BM25 + Vector) for Wikipedia search."""

    def __init__(self):
        self.bm25 = None
        self.documents = []  # Stores title, URL, and snippet for each document

        # Initialize ChromaDB for vector search
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

        # Use lightweight embedding model (all-MiniLM-L6-v2)
        # Fast on CPU, good quality for semantic search
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = None  # Will be initialized when loading/building index

    def load_or_build(self):
        """Load existing index or build a new one if not found."""
        if os.path.exists(INDEX_PATH):
            print(f"Loading BM25 index from {INDEX_PATH}...", file=sys.stderr)
            with open(INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.documents = data["documents"]
            print(f"BM25 index loaded. {len(self.documents)} documents available.", file=sys.stderr)

            # Load or create ChromaDB collection
            print(f"Loading vector index from {CHROMA_PATH}...", file=sys.stderr)
            try:
                self.collection = self.chroma_client.get_collection(
                    name="wikipedia",
                    embedding_function=self.emb_fn
                )
                print(f"Vector index loaded. {self.collection.count()} documents in vector store.", file=sys.stderr)
            except Exception as e:
                print(f"Vector index not found, will be built on next index build: {e}", file=sys.stderr)
                self.collection = None
        else:
            self.build_index()

    def build_index(self):
        """Build hybrid search index (BM25 + Vector) from Wikipedia dataset."""
        # Get subset size from environment variable or use default
        subset_size = int(os.environ.get("WIKI_SUBSET_SIZE", DEFAULT_SUBSET_SIZE))

        print(f"Downloading dataset (English Wikipedia, {subset_size:,} articles)...", file=sys.stderr)
        # Use subset for practical memory usage
        # Set WIKI_SUBSET_SIZE=0 for full dataset (requires ~64GB RAM)
        if subset_size > 0:
            split = f"train[:{subset_size}]"
        else:
            split = "train"
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split=split)

        print("Tokenizing corpus for BM25...", file=sys.stderr)
        tokenized_corpus = []
        for row in tqdm(ds, desc="Processing documents", file=sys.stderr):
            text = row['text']
            # Simple space-based tokenization (consider NLTK or SpaCy for production)
            tokens = text.lower().split()
            tokenized_corpus.append(tokens)

            # Store metadata for search results
            self.documents.append({
                "title": row['title'],
                "url": row['url'],
                "text": text[:500] + "..." if len(text) > 500 else text  # Store snippet
            })

        print("Building BM25 index...", file=sys.stderr)
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"Saving BM25 index to {INDEX_PATH}...", file=sys.stderr)
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": self.bm25, "documents": self.documents}, f)
        print(f"BM25 index build complete. {len(self.documents)} documents indexed.", file=sys.stderr)

        # Build vector index with ChromaDB
        print("Building vector index with ChromaDB...", file=sys.stderr)
        print("(This may take a while for embedding generation)", file=sys.stderr)

        # Create or recreate collection
        try:
            self.chroma_client.delete_collection(name="wikipedia")
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name="wikipedia",
            embedding_function=self.emb_fn
        )

        # Add documents to ChromaDB in batches
        batch_size = 100
        ids = [doc['url'] for doc in self.documents]
        docs = [doc['text'] for doc in self.documents]
        metadatas = [{"title": doc['title'], "url": doc['url']} for doc in self.documents]

        for i in tqdm(range(0, len(docs), batch_size), desc="Embedding documents", file=sys.stderr):
            batch_ids = ids[i:i+batch_size]
            batch_docs = docs[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]

            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metadatas
            )

        print(f"Vector index build complete. {self.collection.count()} documents in vector store.", file=sys.stderr)

    def build_vector_index(self):
        """Build only the vector index from existing BM25 documents.

        Use this when BM25 index exists but vector index is missing.
        Requires BM25 index to be loaded first (call load_or_build()).
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_or_build() first to load BM25 index.")

        print(f"Building vector index from {len(self.documents)} existing documents...", file=sys.stderr)
        print("(This may take a while for embedding generation)", file=sys.stderr)

        # Create or recreate collection
        try:
            self.chroma_client.delete_collection(name="wikipedia")
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name="wikipedia",
            embedding_function=self.emb_fn
        )

        # Add documents to ChromaDB in batches
        batch_size = 100
        ids = [doc['url'] for doc in self.documents]
        docs = [doc['text'] for doc in self.documents]
        metadatas = [{"title": doc['title'], "url": doc['url']} for doc in self.documents]

        for i in tqdm(range(0, len(docs), batch_size), desc="Embedding documents", file=sys.stderr):
            batch_ids = ids[i:i+batch_size]
            batch_docs = docs[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]

            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metadatas
            )

        print(f"Vector index build complete. {self.collection.count()} documents in vector store.", file=sys.stderr)

    def search(self, query: str, top_k: int = 3):
        """
        Search the index using BM25 algorithm (keyword search).

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of document dictionaries with title, url, and text
        """
        if not self.bm25:
            raise ValueError("Index not loaded. Call load_or_build() first.")

        tokenized_query = query.lower().split()
        # Get top N documents based on BM25 scores
        docs = self.bm25.get_top_n(tokenized_query, self.documents, n=top_k)
        return docs

    def vector_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search using vector similarity (semantic search).

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of document dictionaries with title, url, and text
        """
        if not self.collection:
            raise ValueError("Vector index not loaded. Call load_or_build() first.")

        # Query ChromaDB for semantic similarity
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        # Convert ChromaDB results to standard format
        docs = []
        if results['documents'] and results['documents'][0]:
            for i, (doc_text, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                docs.append({
                    "title": metadata.get('title', 'Unknown'),
                    "url": metadata.get('url', ''),
                    "text": doc_text
                })

        return docs

    def hybrid_search(self, query: str, top_k: int = 3, strategy: Literal["keyword", "semantic", "hybrid"] = "hybrid") -> List[Dict]:
        """
        Perform hybrid search combining BM25 and vector search.

        Args:
            query: Search query string
            top_k: Number of top results to return
            strategy: Search strategy - 'keyword' (BM25 only), 'semantic' (vector only), or 'hybrid' (both)

        Returns:
            List of document dictionaries with title, url, and text
        """
        results = []
        seen_urls = set()

        if strategy in ["keyword", "hybrid"]:
            # Get BM25 results
            try:
                bm25_results = self.search(query, top_k=top_k)
                for doc in bm25_results:
                    if doc['url'] not in seen_urls:
                        results.append(doc)
                        seen_urls.add(doc['url'])
            except ValueError as e:
                print(f"BM25 search failed: {e}", file=sys.stderr)

        if strategy in ["semantic", "hybrid"]:
            # Get vector search results
            try:
                vector_results = self.vector_search(query, top_k=top_k)
                for doc in vector_results:
                    if doc['url'] not in seen_urls:
                        results.append(doc)
                        seen_urls.add(doc['url'])
            except ValueError as e:
                print(f"Vector search failed: {e}", file=sys.stderr)

        # Return top_k results (may be more if both searches returned unique results)
        # For hybrid, we interleave results from both methods
        return results[:top_k * 2 if strategy == "hybrid" else top_k]
