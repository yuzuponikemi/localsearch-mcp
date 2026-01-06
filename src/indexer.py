"""
Multi-Source Hybrid Search Indexer for Local Search MCP Server
Supports Wikipedia (static, large) and Local Files (dynamic, personal).
Handles downloading, tokenizing, and indexing data with BM25 + Vector search.
"""
import os
import pickle
import sys
from typing import List, Dict, Literal
from collections import Counter
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from src.chunking import ChunkingStrategy, get_config_for_file, get_smart_config
from src.document_analyzer import DocumentAnalyzer
from src.content_cleaner import ContentCleaner
from src.quality_metrics import QualityAnalyzer
from src.logger import logger

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
            model_name="intfloat/multilingual-e5-large"
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
            extensions: File extensions to include (default: [".md", ".txt", ".py"])
        """
        super().__init__(collection_name="local_files", chroma_path=LOCAL_CHROMA_PATH)
        self.directory_path = directory_path
        self.extensions = extensions if extensions else [".md", ".txt", ".py"]
        # チャンキング戦略クラスを初期化
        self.chunker = ChunkingStrategy()

    def build_index(self):
        """
        Build/rebuild hybrid index from local files with Intelligent Preprocessing Pipeline.

        Pipeline stages:
        1. Document Analysis (quality scoring, language detection)
        2. Smart Chunking (adaptive sizing, language-aware)
        3. Content Cleaning (deduplication, boilerplate removal)
        4. Quality Metrics (size distribution, uniqueness, diversity)
        5. Vector Indexing with structured logging
        """
        # Import here to avoid circular dependency
        from src.loaders import load_local_files

        logger.info(f"Scanning local files in {self.directory_path}")
        raw_documents = load_local_files(self.directory_path, self.extensions)

        if not raw_documents:
            logger.warning(f"No documents found in {self.directory_path}")
            self.bm25 = None
            return

        logger.info(f"Found {len(raw_documents)} local files", total_files=len(raw_documents))

        # Initialize pipeline components
        analyzer = DocumentAnalyzer()
        cleaner = ContentCleaner()
        quality_analyzer = QualityAnalyzer()

        # 1. BM25用: ファイル単位でインデックス（従来通り）
        self.documents = raw_documents
        tokenized_corpus = [doc['text'].lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Document Analysis & Smart Chunking
        logger.info("Stage 1/4: Analyzing documents and applying smart chunking")

        # ChromaDBコレクションの再作成
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            embedding_function=self.emb_fn
        )

        all_chunks = []
        doc_analyses = []
        language_counts = Counter()

        # Process each document with progress tracking
        for idx, doc in enumerate(tqdm(self.documents, desc="Analyzing & Chunking", file=sys.stderr)):
            # Analyze document
            analysis = analyzer.analyze(doc['text'], doc.get('path', ''))
            doc_analyses.append(analysis)
            language_counts[analysis.language] += 1

            # Log progress
            if idx % 10 == 0:
                logger.log_progress(
                    stage="document_analysis",
                    current=idx + 1,
                    total=len(self.documents),
                    metrics={
                        "avg_quality": f"{sum(a.quality_score for a in doc_analyses) / len(doc_analyses):.3f}",
                        "languages_detected": len(language_counts)
                    }
                )

            # Smart chunking with language-aware sizing
            lc_doc = Document(
                page_content=doc['text'],
                metadata={
                    "title": doc['title'],
                    "url": doc['url'],
                    "path": doc.get('path', ''),
                    "quality_score": analysis.quality_score,
                    "language": analysis.language
                }
            )

            # Get smart config (language-aware, content-aware)
            config = get_smart_config(
                filename=doc.get('path', 'unknown.txt'),
                text_content=doc['text']
            )

            # Apply chunking
            chunks = self.chunker.chunk_documents([lc_doc], config)

            # Add chunk index to metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_index'] = i
                chunk.metadata['chunking_method'] = config.method.value
                chunk.metadata['detected_language'] = config.detected_language
                chunk.metadata['language_multiplier'] = str(config.language_multiplier)

            all_chunks.extend(chunks)

        logger.info(
            f"Chunking complete",
            total_chunks_before_cleaning=len(all_chunks),
            avg_chunks_per_doc=f"{len(all_chunks)/len(self.documents):.1f}"
        )

        # 3. Content Cleaning
        logger.info("Stage 2/4: Cleaning content (deduplication, boilerplate removal)")

        cleaned_chunks, cleaning_stats = cleaner.clean_chunks(all_chunks, detect_boilerplate=True)

        logger.info(
            "Content cleaning complete",
            exact_duplicates_removed=cleaning_stats.exact_duplicates_removed,
            near_duplicates_removed=cleaning_stats.near_duplicates_removed,
            boilerplate_removed=cleaning_stats.boilerplate_removed,
            too_small_removed=cleaning_stats.too_small_removed,
            uniqueness_ratio=f"{cleaning_stats.uniqueness_ratio:.3f}",
            chunks_remaining=cleaning_stats.total_output
        )

        # 4. Quality Metrics
        logger.info("Stage 3/4: Computing quality metrics")

        metrics = quality_analyzer.analyze(cleaned_chunks)

        logger.info("Quality metrics computed", **metrics.to_dict())

        # 5. Vector Indexing with Progress
        logger.info("Stage 4/4: Indexing chunks to vector database")

        chunk_ids = []
        chunk_texts = []
        chunk_metadatas = []

        for i, chunk in enumerate(cleaned_chunks):
            chunk_id = f"{chunk.metadata.get('url', 'unknown')}#chunk{chunk.metadata.get('chunk_index', i)}"
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk.page_content)

            # Clean metadata for ChromaDB (no None values)
            meta = chunk.metadata.copy()
            for k, v in meta.items():
                if v is None:
                    meta[k] = ""
                elif not isinstance(v, (str, int, float, bool)):
                    meta[k] = str(v)

            chunk_metadatas.append(meta)

        # Batch insert with progress logging
        if chunk_texts:
            batch_size = 50
            total_batches = (len(chunk_texts) + batch_size - 1) // batch_size

            for batch_idx, i in enumerate(range(0, len(chunk_texts), batch_size)):
                end = min(i + batch_size, len(chunk_texts))

                self.collection.add(
                    ids=chunk_ids[i:end],
                    documents=chunk_texts[i:end],
                    metadatas=chunk_metadatas[i:end]
                )

                # Log progress every 10 batches
                if batch_idx % 10 == 0:
                    logger.log_progress(
                        stage="vector_indexing",
                        current=end,
                        total=len(chunk_texts),
                        metrics={"batch_size": batch_size}
                    )

        # Final summary
        avg_quality = sum(a.quality_score for a in doc_analyses) / len(doc_analyses) if doc_analyses else 0.0

        logger.log_document_stats(
            total_docs=len(self.documents),
            total_chunks=len(chunk_texts),
            avg_quality=avg_quality,
            unique_ratio=cleaning_stats.uniqueness_ratio,
            languages=dict(language_counts)
        )

        logger.info(
            "Local file index build complete",
            files=len(self.documents),
            chunks=len(chunk_texts),
            avg_quality_score=f"{avg_quality:.3f}"
        )
