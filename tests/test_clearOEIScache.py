import os
import pytest
from pathlib import Path
from mapFolding.clearOEIScache import clearOEIScache
from mapFolding import pathCache

def test_clearOEIScache(tmp_path):
    global pathCache
    
    # Create dummy cache files
    test_cache = tmp_path / "test_cache"
    test_cache.mkdir()
    (test_cache / "A000001.txt").write_text("test")
    
    # Replace cache path temporarily
    original_path = pathCache
    try:
        pathCache = test_cache
        clearOEIScache()
        assert not list(test_cache.glob("*.txt"))
    finally:
        pathCache = original_path