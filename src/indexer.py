"""
Multi-Source Hybrid Search Indexer for Local Search MCP Server
Supports Wikipedia (static, large) and Local Files (dynamic, personal).
Handles downloading, tokenizing, and indexing data with BM25 + Vector search.
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

# Default paths for Wikipedia index
WIKI_INDEX_PATH = "data/wiki_index.pkl"
WIKI_CHROMA_PATH = "data/chroma_db"
LOCAL_CHROMA_PATH = "data/local_chroma_db"

# Number of documents to index for Wikipedia
DEFAULT_SUBSET_SIZE = 1_000_000


class BaseHybridIndexer:
    """
    Base class for hybrid search (BM25 + Vector) indexing.
    Provides common functionality for both Wikipedia and Local File indexers.
    """

    def __init__(self, collection_name: str, chroma_path: str):
        """
        Initialize the base indexer.

        Args:
            collection_name: Name for the ChromaDB collection
            chroma_path: Path to the ChromaDB persistent storage
        """
        self.bm25 = None
        self.documents = []  # Stores document metadata

        # Initialize ChromaDB for vector search
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        # Use lightweight embedding model (all-MiniLM-L6-v2)
        # Fast on CPU, good quality for semantic search
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection_name = collection_name
        self.collection = None

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search the index using BM25 algorithm (keyword search).

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of document dictionaries
        """
        if not self.bm25:
            return []

        tokenized_query = query.lower().split()
        # Get top N documents based on BM25 scores
        docs = self.bm25.get_top_n(tokenized_query, self.documents, n=top_k)
        return docs if docs else []

    def vector_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search using vector similarity (semantic search).

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of document dictionaries
        """
        if not self.collection:
            return []

        try:
            # Query ChromaDB for semantic similarity
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

            # Convert ChromaDB results to standard format
            docs = []
            if results['documents'] and results['documents'][0]:
                for doc_text, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    docs.append({
                        "title": metadata.get('title', 'Unknown'),
                        "url": metadata.get('url', ''),
                        "text": doc_text
                    })

            return docs
        except Exception as e:
            print(f"Vector search error: {e}", file=sys.stderr)
            return []

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        strategy: Literal["keyword", "semantic", "hybrid"] = "hybrid"
    ) -> List[Dict]:
        """
        Perform hybrid search combining BM25 and vector search using Reciprocal Rank Fusion (RRF).

        Args:
            query: Search query string
            top_k: Number of top results to return
            strategy: Search strategy - 'keyword' (BM25 only), 'semantic' (vector only), or 'hybrid' (both)

        Returns:
            List of document dictionaries with title, url, text, and source (bm25/vector/both)
        """
        # RRF constant (typical value is 60)
        RRF_K = 60

        # Store RRF scores and sources for each document
        rrf_scores: Dict[str, float] = {}
        doc_data: Dict[str, Dict] = {}
        doc_sources: Dict[str, List[str]] = {}

        if strategy in ["keyword", "hybrid"]:
            bm25_results = self.search(query, top_k=top_k)
            for rank, doc in enumerate(bm25_results):
                url = doc['url']
                rrf_scores[url] = rrf_scores.get(url, 0) + 1 / (RRF_K + rank + 1)
                # Keep the document with longer text (BM25 has full text)
                if url not in doc_data or len(doc.get('text', '')) > len(doc_data[url].get('text', '')):
                    doc_data[url] = doc
                if url not in doc_sources:
                    doc_sources[url] = []
                doc_sources[url].append("bm25")

        if strategy in ["semantic", "hybrid"]:
            vector_results = self.vector_search(query, top_k=top_k)
            for rank, doc in enumerate(vector_results):
                url = doc['url']
                rrf_scores[url] = rrf_scores.get(url, 0) + 1 / (RRF_K + rank + 1)
                # Keep the document with longer text (prefer BM25's full text over vector's truncated text)
                if url not in doc_data or len(doc.get('text', '')) > len(doc_data[url].get('text', '')):
                    doc_data[url] = doc
                if url not in doc_sources:
                    doc_sources[url] = []
                doc_sources[url].append("vector")

        # Sort by RRF score (descending) and take top_k
        sorted_urls = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        # Build final results with source information
        results = []
        for url in sorted_urls:
            doc = doc_data[url].copy()
            sources = doc_sources[url]
            doc['source'] = "both" if len(sources) > 1 else sources[0]
            results.append(doc)

        return results


class WikiIndexer(BaseHybridIndexer):
    """
    Wikipedia indexer with persistent BM25 and vector indices.
    Builds index once and loads from cache on subsequent runs.
    """

    def __init__(self):
        super().__init__(collection_name="wikipedia", chroma_path=WIKI_CHROMA_PATH)
        self.index_path = WIKI_INDEX_PATH

    def load_or_build(self):
        """Load existing index or build a new one if not found."""
        if os.path.exists(self.index_path):
            print(f"Loading BM25 index from {self.index_path}...", file=sys.stderr)
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.documents = data["documents"]
            print(f"BM25 index loaded. {len(self.documents)} documents available.", file=sys.stderr)

            # Load or create ChromaDB collection
            print(f"Loading vector index from {WIKI_CHROMA_PATH}...", file=sys.stderr)
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name,
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
        if subset_size > 0:
            split = f"train[:{subset_size}]"
        else:
            split = "train"
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split=split)

        print("Tokenizing corpus for BM25...", file=sys.stderr)
        tokenized_corpus = []
        for row in tqdm(ds, desc="Processing documents", file=sys.stderr):
            text = row['text']
            # Simple space-based tokenization
            tokens = text.lower().split()
            tokenized_corpus.append(tokens)

            # Store metadata for search results
            self.documents.append({
                "title": row['title'],
                "url": row['url'],
                "text": text[:500] + "..." if len(text) > 500 else text
            })

        print("Building BM25 index...", file=sys.stderr)
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"Saving BM25 index to {self.index_path}...", file=sys.stderr)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "documents": self.documents}, f)
        print(f"BM25 index build complete. {len(self.documents)} documents indexed.", file=sys.stderr)

        # Build vector index with ChromaDB
        print("Building vector index with ChromaDB...", file=sys.stderr)
        print("(This may take a while for embedding generation)", file=sys.stderr)

        # Create or recreate collection
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
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
        """
        Build only the vector index from existing BM25 documents.
        Use this when BM25 index exists but vector index is missing.
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_or_build() first to load BM25 index.")

        print(f"Building vector index from {len(self.documents)} existing documents...", file=sys.stderr)
        print("(This may take a while for embedding generation)", file=sys.stderr)

        # Create or recreate collection
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
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


class LocalFileIndexer(BaseHybridIndexer):
    """
    Local file indexer that scans directories for Markdown/text files.
    Rebuilds BM25 index on each startup (fast for local files).
    Uses persistent vector index with upsert for updates.
    """

    def __init__(self, directory_path: str, extensions: List[str] = None):
        """
        Initialize local file indexer.

        Args:
            directory_path: Directory to scan for files
            extensions: File extensions to include (default: [".md", ".txt"])
        """
        super().__init__(collection_name="local_files", chroma_path=LOCAL_CHROMA_PATH)
        self.directory_path = directory_path
        self.extensions = extensions if extensions else [".md", ".txt"]

    def build_index(self):
        """
        Build/rebuild hybrid index from local files.
        BM25 is rebuilt from scratch (fast for local files).
        Vector index is updated with upsert.
        """
        # Import here to avoid circular dependency
        try:
            from .loaders import load_local_files
        except ImportError:
            from loaders import load_local_files

        print(f"Scanning local files in {self.directory_path}...", file=sys.stderr)
        self.documents = load_local_files(self.directory_path, self.extensions)

        if not self.documents:
            print(f"No documents found in {self.directory_path}", file=sys.stderr)
            self.bm25 = None
            return

        print(f"Found {len(self.documents)} local files", file=sys.stderr)

        # Build BM25 index (in-memory, rebuilt each time)
        print("Building BM25 index for local files...", file=sys.stderr)
        tokenized_corpus = [doc['text'].lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Build/update vector index
        print("Updating vector index for local files...", file=sys.stderr)

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.emb_fn
            )
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.emb_fn
            )

        # Upsert documents in batches (ChromaDB will update existing IDs)
        # Use full text for vector indexing (with reasonable limit for embedding model)
        # all-MiniLM-L6-v2 can handle ~256 tokens, approximately 1500-2000 chars
        MAX_CHARS_FOR_EMBEDDING = 2000
        batch_size = 100
        ids = [doc['url'] for doc in self.documents]
        docs = [
            doc['text'][:MAX_CHARS_FOR_EMBEDDING] + ("..." if len(doc['text']) > MAX_CHARS_FOR_EMBEDDING else "")
            for doc in self.documents
        ]
        metadatas = [{"title": doc['title'], "url": doc['url'], "path": doc.get('path', '')} for doc in self.documents]

        for i in range(0, len(docs), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = docs[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metadatas
            )

        print(f"Local file index build complete. {len(self.documents)} files indexed.", file=sys.stderr)
