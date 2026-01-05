"""
Wikipedia BM25 Indexer for Local Search MCP Server
Handles downloading, tokenizing, and indexing Wikipedia data.
"""
import os
import pickle
import sys
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm

INDEX_PATH = "data/wiki_index.pkl"

# Number of documents to index. Set via WIKI_SUBSET_SIZE env var.
# Default: 1,000,000 (sufficient for HotPotQA evaluation)
# Full English Wikipedia: ~6.4M articles
# For development/testing: 10,000-50,000
DEFAULT_SUBSET_SIZE = 1_000_000

class WikiIndexer:
    """Builds and manages BM25 index for Wikipedia search."""

    def __init__(self):
        self.bm25 = None
        self.documents = []  # Stores title, URL, and snippet for each document

    def load_or_build(self):
        """Load existing index or build a new one if not found."""
        if os.path.exists(INDEX_PATH):
            print(f"Loading index from {INDEX_PATH}...", file=sys.stderr)
            with open(INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.documents = data["documents"]
            print(f"Index loaded. {len(self.documents)} documents available.", file=sys.stderr)
        else:
            self.build_index()

    def build_index(self):
        """Build BM25 index from Wikipedia dataset."""
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

        print("Tokenizing corpus...", file=sys.stderr)
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

        print(f"Saving index to {INDEX_PATH}...", file=sys.stderr)
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": self.bm25, "documents": self.documents}, f)
        print(f"Index build complete. {len(self.documents)} documents indexed.", file=sys.stderr)

    def search(self, query: str, top_k: int = 3):
        """
        Search the index using BM25 algorithm.

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
