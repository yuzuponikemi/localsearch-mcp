"""
Advanced Chunking Module for Local Search MCP
Handles intelligent text splitting for Markdown and Code.
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

class ChunkingMethod(Enum):
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    CODE = "code"

@dataclass
class ChunkingConfig:
    method: ChunkingMethod
    chunk_size: int = 1000
    chunk_overlap: int = 200
    headers_to_split_on: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("#", "h1"), ("##", "h2"), ("###", "h3")
    ])
    language: Optional[Language] = None

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

# Helper function to easily get config
def get_config_for_file(filename: str) -> ChunkingConfig:
    """Determine the best config based on file extension."""
    ext = filename.lower().split('.')[-1]

    if ext in ['md', 'markdown']:
        return ChunkingConfig(method=ChunkingMethod.MARKDOWN, chunk_size=1200, chunk_overlap=200)
    elif ext == 'py':
        return ChunkingConfig(method=ChunkingMethod.CODE, language=Language.PYTHON, chunk_size=1000)
    elif ext in ['js', 'ts']:
        return ChunkingConfig(method=ChunkingMethod.CODE, language=Language.JS, chunk_size=1000)
    else:
        return ChunkingConfig(method=ChunkingMethod.RECURSIVE, chunk_size=1000, chunk_overlap=200)
