"""
Content Cleaner Module for Local Search MCP
Provides deduplication, boilerplate removal, and size filtering for chunks.
"""
import hashlib
import re
from typing import List, Set, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
from langchain_core.documents import Document


@dataclass
class CleaningStats:
    """Statistics from content cleaning process."""
    total_input: int
    exact_duplicates_removed: int
    near_duplicates_removed: int
    boilerplate_removed: int
    too_small_removed: int
    total_output: int

    @property
    def uniqueness_ratio(self) -> float:
        """Calculate uniqueness ratio (0-1)."""
        if self.total_input == 0:
            return 1.0
        return self.total_output / self.total_input


class ContentCleaner:
    """
    Cleans content by removing duplicates, boilerplate, and too-small chunks.
    """

    # Configuration
    MIN_CHUNK_SIZE = 100  # Minimum characters per chunk
    NEAR_DUPLICATE_THRESHOLD = 0.95  # 95% similarity threshold
    BOILERPLATE_MIN_OCCURRENCES = 3  # Minimum occurrences to be considered boilerplate

    def __init__(self):
        self.seen_hashes: Set[str] = set()
        self.seen_contents: List[str] = []
        self.boilerplate_patterns: List[str] = []

    def clean_chunks(
        self,
        chunks: List[Document],
        detect_boilerplate: bool = True
    ) -> Tuple[List[Document], CleaningStats]:
        """
        Clean a list of chunks by removing duplicates and low-quality content.

        Args:
            chunks: List of Document chunks
            detect_boilerplate: Whether to detect and remove boilerplate patterns

        Returns:
            Tuple of (cleaned_chunks, cleaning_stats)
        """
        stats = CleaningStats(
            total_input=len(chunks),
            exact_duplicates_removed=0,
            near_duplicates_removed=0,
            boilerplate_removed=0,
            too_small_removed=0,
            total_output=0
        )

        # Reset state
        self.seen_hashes.clear()
        self.seen_contents.clear()

        # Detect boilerplate patterns if requested
        if detect_boilerplate:
            self.boilerplate_patterns = self._detect_boilerplate_patterns(chunks)

        cleaned_chunks = []

        for chunk in chunks:
            content = chunk.page_content.strip()

            # Skip empty chunks
            if not content:
                stats.too_small_removed += 1
                continue

            # Check minimum size
            if len(content) < self.MIN_CHUNK_SIZE:
                stats.too_small_removed += 1
                continue

            # Check for exact duplicates (MD5 hash)
            content_hash = self._hash_content(content)
            if content_hash in self.seen_hashes:
                stats.exact_duplicates_removed += 1
                continue

            # Check for near duplicates
            if self._is_near_duplicate(content):
                stats.near_duplicates_removed += 1
                continue

            # Check for boilerplate
            if detect_boilerplate and self._is_boilerplate(content):
                stats.boilerplate_removed += 1
                continue

            # Chunk passed all checks
            self.seen_hashes.add(content_hash)
            self.seen_contents.append(content)
            cleaned_chunks.append(chunk)

        stats.total_output = len(cleaned_chunks)
        return cleaned_chunks, stats

    def _hash_content(self, content: str) -> str:
        """Generate MD5 hash of content for exact duplicate detection."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _is_near_duplicate(self, content: str) -> bool:
        """
        Check if content is a near duplicate of previously seen content.
        Uses SequenceMatcher for similarity comparison.
        """
        # Only check against recent contents to avoid O(n²) complexity
        # Check last 100 items max
        check_limit = min(100, len(self.seen_contents))
        recent_contents = self.seen_contents[-check_limit:]

        for seen_content in recent_contents:
            similarity = SequenceMatcher(None, content, seen_content).ratio()
            if similarity >= self.NEAR_DUPLICATE_THRESHOLD:
                return True

        return False

    def _detect_boilerplate_patterns(self, chunks: List[Document]) -> List[str]:
        """
        Detect boilerplate patterns by finding frequently repeated content.

        Returns:
            List of boilerplate pattern strings
        """
        # Count line occurrences
        line_counts = {}
        for chunk in chunks:
            lines = chunk.page_content.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 20:  # Only consider substantial lines
                    line_counts[line] = line_counts.get(line, 0) + 1

        # Find patterns that occur frequently
        boilerplate = []
        for line, count in line_counts.items():
            if count >= self.BOILERPLATE_MIN_OCCURRENCES:
                boilerplate.append(line)

        return boilerplate

    def _is_boilerplate(self, content: str) -> bool:
        """
        Check if content matches boilerplate patterns.
        """
        if not self.boilerplate_patterns:
            return False

        # Check if content is entirely boilerplate
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return False

        boilerplate_lines = sum(
            1 for line in lines
            if any(pattern in line for pattern in self.boilerplate_patterns)
        )

        # Consider it boilerplate if >80% of lines match patterns
        return (boilerplate_lines / len(lines)) > 0.8

    def remove_common_boilerplate(self, text: str) -> str:
        """
        Remove common boilerplate patterns from text.

        This includes:
        - Copyright notices
        - License headers
        - Common footers
        - Navigation elements
        """
        # Remove copyright notices
        text = re.sub(
            r'Copyright\s+©?\s*\d{4}.*?(?=\n\n|\Z)',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Remove license headers (common patterns)
        text = re.sub(
            r'Licensed under.*?(?=\n\n|\Z)',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Remove common navigation patterns
        text = re.sub(
            r'^\s*(?:Home|Back|Next|Previous|Table of Contents)\s*$',
            '',
            text,
            flags=re.MULTILINE | re.IGNORECASE
        )

        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def reset(self):
        """Reset cleaner state (seen hashes and contents)."""
        self.seen_hashes.clear()
        self.seen_contents.clear()
        self.boilerplate_patterns.clear()
