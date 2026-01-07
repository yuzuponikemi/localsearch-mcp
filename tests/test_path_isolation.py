"""
Test that different LOCAL_DOCS_PATH values use isolated ChromaDB collections.

This ensures that switching between different document directories does not
cause data mixing in the vector store.
"""
import os
import tempfile
import shutil
from pathlib import Path
from src.indexer import LocalFileIndexer


def test_different_paths_use_different_collections():
    """Test that two different paths get different collection names."""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        # Create test files in both directories
        Path(tmpdir1, "test1.md").write_text("# Test Document 1\nContent for path 1")
        Path(tmpdir2, "test2.md").write_text("# Test Document 2\nContent for path 2")
        
        # Create indexers for each path
        indexer1 = LocalFileIndexer(tmpdir1)
        indexer2 = LocalFileIndexer(tmpdir2)
        
        # Verify they use different collection names
        assert indexer1.collection_name != indexer2.collection_name, \
            f"Different paths should use different collections: {indexer1.collection_name} vs {indexer2.collection_name}"
        
        # Verify collection names include path hash
        assert indexer1.collection_name.startswith("local_files_"), \
            f"Collection name should start with 'local_files_': {indexer1.collection_name}"
        assert indexer2.collection_name.startswith("local_files_"), \
            f"Collection name should start with 'local_files_': {indexer2.collection_name}"
        
        print(f"✅ Path 1 ({tmpdir1[:30]}...) -> Collection: {indexer1.collection_name}")
        print(f"✅ Path 2 ({tmpdir2[:30]}...) -> Collection: {indexer2.collection_name}")


def test_same_path_uses_same_collection():
    """Test that the same path (accessed twice) uses the same collection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "test.md").write_text("# Test Document\nSome content")
        
        # Create two indexers for the same path
        indexer1 = LocalFileIndexer(tmpdir)
        indexer2 = LocalFileIndexer(tmpdir)
        
        # Verify they use the same collection name
        assert indexer1.collection_name == indexer2.collection_name, \
            f"Same path should use same collection: {indexer1.collection_name} vs {indexer2.collection_name}"
        
        print(f"✅ Same path uses consistent collection: {indexer1.collection_name}")


def test_path_normalization():
    """Test that different representations of the same path use the same collection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "test.md").write_text("# Test Document\nSome content")
        
        # Create indexers with different path representations (trailing slash, etc.)
        path1 = tmpdir
        path2 = tmpdir + os.sep  # Add trailing separator
        path3 = os.path.abspath(tmpdir)  # Explicit absolute path
        
        indexer1 = LocalFileIndexer(path1)
        indexer2 = LocalFileIndexer(path2)
        indexer3 = LocalFileIndexer(path3)
        
        # All should normalize to the same collection
        assert indexer1.collection_name == indexer2.collection_name == indexer3.collection_name, \
            f"Path normalization failed: {indexer1.collection_name}, {indexer2.collection_name}, {indexer3.collection_name}"
        
        print(f"✅ Path normalization works: {indexer1.collection_name}")


def test_state_file_isolation():
    """Test that different paths use different state files."""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        Path(tmpdir1, "test1.md").write_text("# Test 1")
        Path(tmpdir2, "test2.md").write_text("# Test 2")
        
        indexer1 = LocalFileIndexer(tmpdir1)
        indexer2 = LocalFileIndexer(tmpdir2)
        
        # Verify different state files
        assert indexer1.state_file_path != indexer2.state_file_path, \
            f"Different paths should use different state files: {indexer1.state_file_path} vs {indexer2.state_file_path}"
        
        # Verify state file paths include path hash
        assert indexer1.path_hash in indexer1.state_file_path, \
            f"State file should include path hash: {indexer1.state_file_path}"
        assert indexer2.path_hash in indexer2.state_file_path, \
            f"State file should include path hash: {indexer2.state_file_path}"
        
        print(f"✅ Path 1 -> State file: {indexer1.state_file_path}")
        print(f"✅ Path 2 -> State file: {indexer2.state_file_path}")


def test_indexing_isolation():
    """Test that indexing different paths doesn't cause data mixing."""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        # Create distinct test files
        Path(tmpdir1, "doc_a.md").write_text("# Document A\nUnique content about apples")
        Path(tmpdir2, "doc_b.md").write_text("# Document B\nUnique content about bananas")
        
        # Index first path
        indexer1 = LocalFileIndexer(tmpdir1)
        indexer1.build_index()
        
        # Verify first indexer has 1 document
        assert len(indexer1.documents) == 1, f"Indexer 1 should have 1 document, got {len(indexer1.documents)}"
        assert "apples" in indexer1.documents[0]['text'].lower(), \
            f"Indexer 1 should contain 'apples'"
        
        # Index second path
        indexer2 = LocalFileIndexer(tmpdir2)
        indexer2.build_index()
        
        # Verify second indexer has 1 document (not 2)
        assert len(indexer2.documents) == 1, f"Indexer 2 should have 1 document, got {len(indexer2.documents)}"
        assert "bananas" in indexer2.documents[0]['text'].lower(), \
            f"Indexer 2 should contain 'bananas'"
        
        # Verify collections are different
        assert indexer1.collection.name != indexer2.collection.name, \
            "Different paths should use different ChromaDB collections"
        
        print(f"✅ Indexer 1 collection '{indexer1.collection.name}' has {len(indexer1.documents)} docs")
        print(f"✅ Indexer 2 collection '{indexer2.collection.name}' has {len(indexer2.documents)} docs")
        print("✅ No data mixing between different paths!")


if __name__ == "__main__":
    print("="*60)
    print("Testing Path-Specific Collection Isolation")
    print("="*60)
    
    print("\n[Test 1] Different paths use different collections")
    test_different_paths_use_different_collections()
    
    print("\n[Test 2] Same path uses same collection")
    test_same_path_uses_same_collection()
    
    print("\n[Test 3] Path normalization")
    test_path_normalization()
    
    print("\n[Test 4] State file isolation")
    test_state_file_isolation()
    
    print("\n[Test 5] Indexing isolation (no data mixing)")
    test_indexing_isolation()
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
