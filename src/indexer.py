"""
Wikipedia BM25 Indexer for Local Search MCP Server
Handles downloading, tokenizing, and indexing Wikipedia data.
"""
import os
import pickle
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm

INDEX_PATH = "data/wiki_index.pkl"

class WikiIndexer:
    """Builds and manages BM25 index for Wikipedia search."""

    def __init__(self):
        self.bm25 = None
        self.documents = []  # Stores title, URL, and snippet for each document

    def load_or_build(self):
        """Load existing index or build a new one if not found."""
        if os.path.exists(INDEX_PATH):
            print(f"Loading index from {INDEX_PATH}...")
            with open(INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.documents = data["documents"]
            print(f"Index loaded. {len(self.documents)} documents available.")
        else:
            self.build_index()

    def build_index(self):
        """Build BM25 index from Wikipedia dataset."""
        print("Downloading dataset (Simple English Wikipedia)...")
        # Using lightweight Simple English Wikipedia for development/testing
        # For production, consider using 'en' or similar
        ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train[:10000]")

        print("Tokenizing corpus...")
        tokenized_corpus = []
        for row in tqdm(ds, desc="Processing documents"):
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

        print("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"Saving index to {INDEX_PATH}...")
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": self.bm25, "documents": self.documents}, f)
        print(f"Index build complete. {len(self.documents)} documents indexed.")

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
