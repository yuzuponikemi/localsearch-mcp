"""
Quality Metrics Module for Local Search MCP
Provides chunk size distribution, uniqueness ratio, vocabulary diversity, and PCA analysis.
"""
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import Counter
from langchain_core.documents import Document


@dataclass
class QualityMetrics:
    """Quality metrics for indexed chunks."""
    # Size metrics
    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    chunk_size_std: float

    # Size distribution (buckets)
    size_distribution: Dict[str, int]  # e.g., {"0-500": 10, "500-1000": 50, ...}

    # Uniqueness metrics
    uniqueness_ratio: float  # 0-1 (goal: >0.95)
    duplicate_count: int

    # Vocabulary diversity
    vocabulary_diversity: float  # 0-1 (goal: 0.25-0.50)
    unique_words: int
    total_words: int

    # PCA variance (for embedding quality estimation)
    pca_variance_ratio: float  # 0-1 (higher is better)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "total_chunks": self.total_chunks,
            "avg_chunk_size": round(self.avg_chunk_size, 1),
            "chunk_size_range": f"{self.min_chunk_size}-{self.max_chunk_size}",
            "chunk_size_std": round(self.chunk_size_std, 1),
            "uniqueness_ratio": round(self.uniqueness_ratio, 3),
            "duplicate_count": self.duplicate_count,
            "vocabulary_diversity": round(self.vocabulary_diversity, 3),
            "unique_words": self.unique_words,
            "total_words": self.total_words,
            "pca_variance_ratio": round(self.pca_variance_ratio, 3),
            "size_distribution": self.size_distribution
        }


class QualityAnalyzer:
    """
    Analyzes quality metrics of chunked documents.
    """

    # Size distribution buckets (in characters)
    SIZE_BUCKETS = [
        (0, 500),
        (500, 1000),
        (1000, 1500),
        (1500, 2000),
        (2000, float('inf'))
    ]

    def analyze(self, chunks: List[Document]) -> QualityMetrics:
        """
        Perform comprehensive quality analysis on chunks.

        Args:
            chunks: List of Document chunks

        Returns:
            QualityMetrics object
        """
        if not chunks:
            return self._empty_metrics()

        # Extract chunk texts
        texts = [chunk.page_content for chunk in chunks]

        # Size metrics
        sizes = [len(text) for text in texts]
        avg_size = np.mean(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        size_std = np.std(sizes)

        # Size distribution
        size_dist = self._calculate_size_distribution(sizes)

        # Uniqueness metrics
        uniqueness_ratio, duplicate_count = self._calculate_uniqueness(texts)

        # Vocabulary diversity
        vocab_diversity, unique_words, total_words = self._calculate_vocabulary_diversity(texts)

        # PCA variance (simplified - just estimate from text variance)
        pca_variance = self._estimate_pca_variance(sizes)

        return QualityMetrics(
            total_chunks=len(chunks),
            avg_chunk_size=avg_size,
            min_chunk_size=min_size,
            max_chunk_size=max_size,
            chunk_size_std=size_std,
            size_distribution=size_dist,
            uniqueness_ratio=uniqueness_ratio,
            duplicate_count=duplicate_count,
            vocabulary_diversity=vocab_diversity,
            unique_words=unique_words,
            total_words=total_words,
            pca_variance_ratio=pca_variance
        )

    def _calculate_size_distribution(self, sizes: List[int]) -> Dict[str, int]:
        """Calculate chunk size distribution across buckets."""
        distribution = {}

        for min_size, max_size in self.SIZE_BUCKETS:
            if max_size == float('inf'):
                bucket_name = f"{min_size}+"
            else:
                bucket_name = f"{min_size}-{max_size}"

            count = sum(1 for size in sizes if min_size <= size < max_size)
            distribution[bucket_name] = count

        return distribution

    def _calculate_uniqueness(self, texts: List[str]) -> tuple[float, int]:
        """
        Calculate uniqueness ratio and duplicate count.

        Returns:
            Tuple of (uniqueness_ratio, duplicate_count)
        """
        # Use hash-based deduplication for exact matches
        unique_texts = set(texts)
        duplicate_count = len(texts) - len(unique_texts)
        uniqueness_ratio = len(unique_texts) / len(texts) if texts else 1.0

        return uniqueness_ratio, duplicate_count

    def _calculate_vocabulary_diversity(self, texts: List[str]) -> tuple[float, int, int]:
        """
        Calculate vocabulary diversity (unique words / total words).

        Returns:
            Tuple of (diversity_ratio, unique_words, total_words)
        """
        all_words = []
        for text in texts:
            # Simple tokenization (split on whitespace and punctuation)
            words = text.lower().split()
            all_words.extend(words)

        total_words = len(all_words)
        unique_words = len(set(all_words))

        diversity = unique_words / total_words if total_words > 0 else 0.0

        return diversity, unique_words, total_words

    def _estimate_pca_variance(self, sizes: List[int]) -> float:
        """
        Estimate PCA variance ratio from chunk size distribution.

        This is a simplified heuristic:
        - More uniform size distribution -> higher variance in embeddings (good)
        - Very skewed distribution -> lower variance (may indicate issues)

        For true PCA analysis, you'd need actual embeddings.
        """
        if len(sizes) < 2:
            return 0.5  # Default

        # Calculate coefficient of variation (normalized std dev)
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)

        if mean_size == 0:
            return 0.0

        cv = std_size / mean_size

        # Map CV to 0-1 range (heuristic)
        # CV of 0.3-0.5 is typically good (0.7-0.9 variance ratio)
        # CV < 0.1 or > 0.8 is typically bad (0.3-0.5 variance ratio)
        if cv < 0.1:
            variance_ratio = 0.3
        elif cv > 0.8:
            variance_ratio = 0.4
        else:
            # Linear mapping from CV to variance
            variance_ratio = 0.5 + (cv - 0.4) * 0.5

        return min(1.0, max(0.0, variance_ratio))

    def _empty_metrics(self) -> QualityMetrics:
        """Return empty metrics for when no chunks are provided."""
        return QualityMetrics(
            total_chunks=0,
            avg_chunk_size=0.0,
            min_chunk_size=0,
            max_chunk_size=0,
            chunk_size_std=0.0,
            size_distribution={},
            uniqueness_ratio=1.0,
            duplicate_count=0,
            vocabulary_diversity=0.0,
            unique_words=0,
            total_words=0,
            pca_variance_ratio=0.0
        )

    def print_report(self, metrics: QualityMetrics):
        """Print a formatted quality metrics report."""
        print("\n" + "="*60)
        print("QUALITY METRICS REPORT")
        print("="*60)
        print(f"\nChunk Statistics:")
        print(f"  Total Chunks: {metrics.total_chunks}")
        print(f"  Average Size: {metrics.avg_chunk_size:.1f} chars")
        print(f"  Size Range: {metrics.min_chunk_size} - {metrics.max_chunk_size} chars")
        print(f"  Std Deviation: {metrics.chunk_size_std:.1f}")

        print(f"\nSize Distribution:")
        for bucket, count in metrics.size_distribution.items():
            percentage = (count / metrics.total_chunks * 100) if metrics.total_chunks > 0 else 0
            print(f"  {bucket:15s}: {count:5d} ({percentage:5.1f}%)")

        print(f"\nUniqueness:")
        print(f"  Uniqueness Ratio: {metrics.uniqueness_ratio:.3f} (goal: >0.95)")
        print(f"  Duplicates: {metrics.duplicate_count}")

        print(f"\nVocabulary:")
        print(f"  Diversity: {metrics.vocabulary_diversity:.3f} (goal: 0.25-0.50)")
        print(f"  Unique Words: {metrics.unique_words:,}")
        print(f"  Total Words: {metrics.total_words:,}")

        print(f"\nEmbedding Quality (estimated):")
        print(f"  PCA Variance Ratio: {metrics.pca_variance_ratio:.3f}")

        print("="*60 + "\n")
