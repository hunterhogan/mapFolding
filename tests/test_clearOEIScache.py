from unittest.mock import patch
import pytest
from mapFolding import clearOEIScache
from mapFolding.oeis import pathCache

@patch('shutil.rmtree')
@patch('pathlib.Path.exists')
@patch('pathlib.Path.mkdir')
def test_clear_existing_cache(mock_mkdir, mock_exists, mock_rmtree):
    # Setup mocks
    mock_exists.return_value = True
    
    # Run the function
    clearOEIScache()
    
    # Verify the expected calls were made
    mock_rmtree.assert_called_once_with(pathCache)
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

@patch('pathlib.Path.exists')
def test_clear_nonexistent_cache(mock_exists):
    # Setup mock to simulate missing cache directory
    mock_exists.return_value = False
    
    # Run the function
    clearOEIScache()
    
    # Verify exists was called but rmtree wasn't needed
    mock_exists.assert_called_once_with()

