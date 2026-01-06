"""
Structured Logging Module for Local Search MCP
Provides JSON-formatted logs with progress tracking for vectorization.
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

# Environment configuration
STRUCTURED_LOGS_JSON = os.environ.get("STRUCTURED_LOGS_JSON", "false").lower() == "true"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


class StructuredLogger:
    """
    Structured logger with JSON output support and progress tracking.
    """

    def __init__(self, name: str = "localsearch-mcp"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

        # Remove existing handlers
        self.logger.handlers = []

        # Add appropriate handler based on configuration
        handler = logging.StreamHandler(sys.stderr)
        if STRUCTURED_LOGS_JSON:
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        self.logger.addHandler(handler)

    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        self._log(logging.ERROR, message, **kwargs)

    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with structured data support."""
        if STRUCTURED_LOGS_JSON:
            # Include extra data in the log record
            self.logger.log(level, message, extra={"data": kwargs})
        else:
            # Format extra data as string for plain text logs
            if kwargs:
                extra_str = " | " + " ".join([f"{k}={v}" for k, v in kwargs.items()])
                self.logger.log(level, message + extra_str)
            else:
                self.logger.log(level, message)

    def log_progress(
        self,
        stage: str,
        current: int,
        total: int,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Log progress for long-running operations like vectorization.

        Args:
            stage: Current stage name (e.g., "chunking", "embedding", "indexing")
            current: Current item number
            total: Total items
            metrics: Optional metrics dictionary
        """
        percentage = (current / total * 100) if total > 0 else 0

        log_data = {
            "stage": stage,
            "progress": f"{current}/{total}",
            "percentage": f"{percentage:.1f}%"
        }

        if metrics:
            log_data.update(metrics)

        self.info(f"Progress: {stage}", **log_data)

    def log_document_stats(
        self,
        total_docs: int,
        total_chunks: int,
        avg_quality: float,
        unique_ratio: float,
        languages: Dict[str, int]
    ):
        """
        Log document processing statistics.

        Args:
            total_docs: Total number of documents processed
            total_chunks: Total number of chunks created
            avg_quality: Average quality score (0-1)
            unique_ratio: Uniqueness ratio (0-1)
            languages: Dictionary of language counts
        """
        self.info(
            "Document processing complete",
            total_documents=total_docs,
            total_chunks=total_chunks,
            avg_chunks_per_doc=f"{total_chunks/total_docs:.1f}" if total_docs > 0 else 0,
            avg_quality_score=f"{avg_quality:.3f}",
            uniqueness_ratio=f"{unique_ratio:.3f}",
            languages=languages
        )


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra data if present
        if hasattr(record, "data") and record.data:
            log_data.update(record.data)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


# Global logger instance
logger = StructuredLogger()
