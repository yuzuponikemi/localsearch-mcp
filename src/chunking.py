"""
Advanced Chunking Module for Local Search MCP
Handles intelligent text splitting for Markdown and Code with smart sizing.
"""
from enum import Enum
from typing import List, Optional, Any, Tuple
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
)

try:
    from .document_analyzer import DocumentAnalyzer
except ImportError:
    from document_analyzer import DocumentAnalyzer

class ChunkingMethod(Enum):
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    CODE = "code"
    HYBRID = "hybrid"  # Markdown + Code awareness

@dataclass
class ChunkingConfig:
    method: ChunkingMethod
    chunk_size: int = 1000
    chunk_overlap: int = 200
    headers_to_split_on: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("#", "h1"), ("##", "h2"), ("###", "h3")
    ])
    language: Optional[Language] = None
    detected_language: str = "en"  # ISO language code (e.g., "en", "ja")
    language_multiplier: float = 1.0  # Chunk size multiplier for language (e.g., 1.2 for Japanese)

class ChunkingStrategy:
    """
    Selects and applies the best chunking strategy based on file type.
    """

    def chunk_documents(self, documents: List[Document], config: ChunkingConfig) -> List[Document]:
        """
        Apply the configured chunking strategy to a list of documents.
        """
        if not documents:
            return []

        if config.method == ChunkingMethod.MARKDOWN:
            return self._chunk_markdown(documents, config)
        elif config.method == ChunkingMethod.CODE:
            return self._chunk_code(documents, config)
        else:
            return self._chunk_recursive(documents, config)

    def _chunk_markdown(self, documents: List[Document], config: ChunkingConfig) -> List[Document]:
        """
        Split by Markdown headers first, then by character count.
        """
        # 1. Header Split
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.headers_to_split_on,
            strip_headers=False
        )

        header_splits = []
        for doc in documents:
            # Markdown分割を実行
            splits = markdown_splitter.split_text(doc.page_content)
            # 元のメタデータ（URLなど）を継承
            for split in splits:
                split.metadata.update(doc.metadata)
            header_splits.extend(splits)

        # 2. Recursive Split (for sections that are still too large)
        return self._chunk_recursive(header_splits, config)

    def _chunk_code(self, documents: List[Document], config: ChunkingConfig) -> List[Document]:
        """
        Split code using language-specific separators.
        """
        lang = config.language or Language.PYTHON
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        return splitter.split_documents(documents)

    def _chunk_recursive(self, documents: List[Document], config: ChunkingConfig) -> List[Document]:
        """
        Standard recursive character splitting.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_documents(documents)

# Helper function to easily get config (deprecated, use get_smart_config)
def get_config_for_file(filename: str) -> ChunkingConfig:
    """Determine the best config based on file extension (legacy)."""
    ext = filename.lower().split('.')[-1]

    if ext in ['md', 'markdown']:
        return ChunkingConfig(method=ChunkingMethod.MARKDOWN, chunk_size=1200, chunk_overlap=200)
    elif ext == 'py':
        return ChunkingConfig(method=ChunkingMethod.CODE, language=Language.PYTHON, chunk_size=1000)
    elif ext in ['js', 'ts']:
        return ChunkingConfig(method=ChunkingMethod.CODE, language=Language.JS, chunk_size=1000)
    else:
        return ChunkingConfig(method=ChunkingMethod.RECURSIVE, chunk_size=1000, chunk_overlap=200)


def get_smart_config(
    filename: str,
    text_content: str,
    target_size_min: int = 500,
    target_size_max: int = 1000
) -> ChunkingConfig:
    """
    Determine the best chunking config with smart sizing based on content analysis.

    Args:
        filename: File path for type detection
        text_content: Document text for language detection
        target_size_min: Target minimum chunk size in characters
        target_size_max: Target maximum chunk size in characters

    Returns:
        ChunkingConfig with optimized settings
    """
    # Analyze document
    analyzer = DocumentAnalyzer()
    analysis = analyzer.analyze(text_content, filename)

    # Determine file extension
    ext = filename.lower().split('.')[-1]

    # Determine base method
    if ext in ['md', 'markdown']:
        method = ChunkingMethod.MARKDOWN
        base_size = target_size_max + 200  # Slightly larger for Markdown
    elif ext == 'py':
        method = ChunkingMethod.CODE
        code_lang = Language.PYTHON
        base_size = target_size_max
    elif ext in ['js', 'ts']:
        method = ChunkingMethod.CODE
        code_lang = Language.JS
        base_size = target_size_max
    else:
        method = ChunkingMethod.RECURSIVE
        code_lang = None
        base_size = target_size_max

    # Language-specific multiplier
    language_multiplier = 1.0
    if analysis.language == "ja":  # Japanese
        language_multiplier = 1.2
    elif analysis.language == "zh":  # Chinese
        language_multiplier = 1.15
    elif analysis.language == "ko":  # Korean
        language_multiplier = 1.1

    # Apply language multiplier
    adjusted_size = int(base_size * language_multiplier)
    overlap = int(adjusted_size * 0.2)  # 20% overlap

    return ChunkingConfig(
        method=method,
        chunk_size=adjusted_size,
        chunk_overlap=overlap,
        language=code_lang if method == ChunkingMethod.CODE else None,
        detected_language=analysis.language,
        language_multiplier=language_multiplier
    )
