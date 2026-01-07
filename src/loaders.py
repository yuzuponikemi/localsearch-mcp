"""
Local File Loaders for Multi-Source Search
Handles loading local files (Markdown, text, etc.) for indexing.
"""
import os
import glob
import sys
from typing import List, Dict


def load_local_files(directory_path: str, extensions: List[str] = None) -> List[Dict]:
    """
    Load local text files from a directory recursively.

    Args:
        directory_path: Path to the directory to scan
        extensions: List of file extensions to include (default: [".md", ".txt"])

    Returns:
        List of document dictionaries with id, text, title, source, and path
    """
    if extensions is None:
        extensions = [".md", ".txt"]

    documents = []

    if not os.path.exists(directory_path):
        print(f"Warning: Directory not found: {directory_path}", file=sys.stderr)
        return []

    if not os.path.isdir(directory_path):
        print(f"Warning: Path is not a directory: {directory_path}", file=sys.stderr)
        return []

    for ext in extensions:
        # Recursively search for files with the given extension
        pattern = os.path.join(directory_path, "**", f"*{ext}")
        files = glob.glob(pattern, recursive=True)

        for file_path in files:
            try:
                # Try multiple encodings to handle various file types
                content = None
                for encoding in ['utf-8', 'utf-8-sig', 'cp932', 'shift-jis', 'iso-8859-1']:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            content = f.read()
                        break  # Successfully read the file
                    except UnicodeDecodeError:
                        continue  # Try next encoding

                if content is None:
                    print(f"Warning: Skipping file {file_path}: Unable to decode with supported encodings", file=sys.stderr)
                    continue

                # Skip empty files
                if not content.strip():
                    continue

                # Calculate relative path for cleaner IDs
                rel_path = os.path.relpath(file_path, directory_path)

                # Create document entry
                documents.append({
                    "id": f"file://{rel_path}",
                    "text": content,
                    "title": os.path.basename(file_path),
                    "source": "local_file",
                    "path": file_path,
                    "url": f"file://{file_path}"  # Compatible with existing result format
                })
            except Exception as e:
                print(f"Warning: Skipping file {file_path}: {e}", file=sys.stderr)

    return documents
