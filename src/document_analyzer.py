"""
Document Analyzer Module for Local Search MCP
Provides quality scoring, language detection, structure analysis, and issue detection.
"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import langdetect
from langdetect import DetectorFactory

# Set seed for consistent language detection
DetectorFactory.seed = 0


class DocumentType(Enum):
    """Document type classification."""
    MARKDOWN = "markdown"
    CODE_PYTHON = "code_python"
    CODE_JAVASCRIPT = "code_javascript"
    CODE_TYPESCRIPT = "code_typescript"
    PLAIN_TEXT = "plain_text"
    UNKNOWN = "unknown"


@dataclass
class DocumentAnalysis:
    """Results of document analysis."""
    quality_score: float  # 0-1 scale
    language: str  # ISO language code (e.g., "en", "ja")
    language_confidence: float  # 0-1 scale
    document_type: DocumentType
    char_count: int
    word_count: int
    line_count: int
    avg_line_length: float
    issues: List[str]  # List of detected issues
    recommendations: List[str]  # List of recommendations


class DocumentAnalyzer:
    """
    Analyzes documents for quality, language, structure, and potential issues.
    """

    # Quality scoring thresholds
    MIN_CONTENT_LENGTH = 50
    IDEAL_AVG_LINE_LENGTH = 80
    MAX_AVG_LINE_LENGTH = 200

    def analyze(self, text: str, file_path: Optional[str] = None) -> DocumentAnalysis:
        """
        Perform comprehensive document analysis.

        Args:
            text: Document text content
            file_path: Optional file path for type detection

        Returns:
            DocumentAnalysis object with all analysis results
        """
        # Basic metrics
        char_count = len(text)
        lines = text.split('\n')
        line_count = len(lines)
        word_count = len(text.split())
        avg_line_length = char_count / line_count if line_count > 0 else 0

        # Document type detection
        doc_type = self._detect_document_type(text, file_path)

        # Language detection
        language, lang_confidence = self._detect_language(text)

        # Issue detection
        issues = self._detect_issues(text, char_count, word_count, avg_line_length)

        # Quality scoring
        quality_score = self._calculate_quality_score(
            text, char_count, word_count, avg_line_length, issues
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            quality_score, issues, doc_type, language
        )

        return DocumentAnalysis(
            quality_score=quality_score,
            language=language,
            language_confidence=lang_confidence,
            document_type=doc_type,
            char_count=char_count,
            word_count=word_count,
            line_count=line_count,
            avg_line_length=avg_line_length,
            issues=issues,
            recommendations=recommendations
        )

    def _detect_document_type(
        self, text: str, file_path: Optional[str] = None
    ) -> DocumentType:
        """Detect document type from content and file extension."""
        if file_path:
            ext = file_path.lower().split('.')[-1]
            if ext in ['md', 'markdown']:
                return DocumentType.MARKDOWN
            elif ext == 'py':
                return DocumentType.CODE_PYTHON
            elif ext == 'js':
                return DocumentType.CODE_JAVASCRIPT
            elif ext == 'ts':
                return DocumentType.CODE_TYPESCRIPT

        # Content-based detection
        if re.search(r'^#{1,6}\s+', text, re.MULTILINE):
            return DocumentType.MARKDOWN
        elif re.search(r'\bdef\s+\w+\s*\(|\bclass\s+\w+\s*[:(]', text):
            return DocumentType.CODE_PYTHON
        elif re.search(r'\bfunction\s+\w+\s*\(|\bconst\s+\w+\s*=', text):
            return DocumentType.CODE_JAVASCRIPT

        return DocumentType.PLAIN_TEXT

    def _detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of the text.

        Returns:
            Tuple of (language_code, confidence)
        """
        # Remove code blocks and URLs for better detection
        clean_text = re.sub(r'```[\s\S]*?```', '', text)
        clean_text = re.sub(r'http[s]?://\S+', '', clean_text)
        clean_text = re.sub(r'`[^`]+`', '', clean_text)

        try:
            # Detect language
            lang = langdetect.detect(clean_text)
            # Get confidence (langdetect doesn't provide confidence directly)
            # We use a heuristic based on text length
            confidence = min(1.0, len(clean_text) / 500)  # Higher confidence for longer texts
            return lang, confidence
        except:
            return "unknown", 0.0

    def _detect_issues(
        self,
        text: str,
        char_count: int,
        word_count: int,
        avg_line_length: float
    ) -> List[str]:
        """Detect potential issues in the document."""
        issues = []

        # Check minimum content length
        if char_count < self.MIN_CONTENT_LENGTH:
            issues.append(f"Content too short ({char_count} chars, minimum {self.MIN_CONTENT_LENGTH})")

        # Check for very long lines
        if avg_line_length > self.MAX_AVG_LINE_LENGTH:
            issues.append(f"Average line length too long ({avg_line_length:.0f} chars)")

        # Check for repetitive content
        lines = text.split('\n')
        unique_lines = set(lines)
        if len(lines) > 10 and len(unique_lines) / len(lines) < 0.5:
            issues.append("High line repetition detected")

        # Check for excessive whitespace
        whitespace_ratio = len(re.findall(r'\s', text)) / len(text) if len(text) > 0 else 0
        if whitespace_ratio > 0.5:
            issues.append(f"Excessive whitespace ({whitespace_ratio*100:.1f}%)")

        # Check for very low word count
        if word_count < 10 and char_count > self.MIN_CONTENT_LENGTH:
            issues.append("Very low word count (possible non-textual content)")

        return issues

    def _calculate_quality_score(
        self,
        text: str,
        char_count: int,
        word_count: int,
        avg_line_length: float,
        issues: List[str]
    ) -> float:
        """
        Calculate document quality score (0-1).

        Factors:
        - Content length
        - Line length appropriateness
        - Issue count
        - Content density
        """
        score = 1.0

        # Penalty for short content
        if char_count < self.MIN_CONTENT_LENGTH:
            score *= char_count / self.MIN_CONTENT_LENGTH

        # Penalty for line length issues
        if avg_line_length > self.MAX_AVG_LINE_LENGTH:
            penalty = (avg_line_length - self.MAX_AVG_LINE_LENGTH) / self.MAX_AVG_LINE_LENGTH
            score *= max(0.5, 1.0 - penalty)

        # Penalty for each issue
        issue_penalty = 0.1 * len(issues)
        score *= max(0.3, 1.0 - issue_penalty)

        # Bonus for good content density (words per character)
        if char_count > 0:
            density = word_count / char_count
            if 0.15 <= density <= 0.25:  # Typical for good text
                score *= 1.1

        return min(1.0, max(0.0, score))

    def _generate_recommendations(
        self,
        quality_score: float,
        issues: List[str],
        doc_type: DocumentType,
        language: str
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if quality_score < 0.5:
            recommendations.append("Consider improving document quality before indexing")

        if "Content too short" in str(issues):
            recommendations.append("Add more content for better search results")

        if "Average line length too long" in str(issues):
            recommendations.append("Break long lines for better readability")

        if "High line repetition detected" in str(issues):
            recommendations.append("Remove repetitive content")

        # Type-specific recommendations
        if doc_type == DocumentType.MARKDOWN and quality_score > 0.7:
            recommendations.append("Use Markdown chunking strategy for best results")
        elif doc_type in [DocumentType.CODE_PYTHON, DocumentType.CODE_JAVASCRIPT, DocumentType.CODE_TYPESCRIPT]:
            recommendations.append("Use code-aware chunking strategy")

        # Language-specific recommendations
        if language == "ja":
            recommendations.append("Japanese detected: chunk size will be adjusted (1.2x)")
        elif language == "unknown":
            recommendations.append("Language detection failed: using default chunking")

        if not recommendations:
            recommendations.append("Document quality is good, ready for indexing")

        return recommendations
