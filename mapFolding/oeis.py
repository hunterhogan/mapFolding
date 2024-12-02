import pathlib
from datetime import datetime, timedelta
import urllib.request
from mapFolding import count_foldings

"""
functional, not OOP, programming
use pathlib for disk I/O. i.e., not `open()`
do not duplicate logic: DRY
"""

def _construct_url(sequence_number: str) -> str:
    return f"https://oeis.org/{sequence_number}/b{sequence_number[1:]}.txt"

def _get_cache_path(sequence_number: str) -> pathlib.Path:
    return pathlib.Path(__file__).parent / ".cache" / f"{sequence_number}.txt"

def _parse_content(content: str, sequence_number: str) -> dict[int, int]:
    lines = content.strip().splitlines()
    if not any(line.startswith(f"# {sequence_number}") for line in lines):
        raise ValueError(f"Content does not match sequence {sequence_number}")
    
    result = {}
    for line in lines:
        if line.startswith('#'):
            continue
        n, value = map(int, line.split())
        result[n] = value
    return result

def _fetch_from_url(url: str) -> str:
    response = urllib.request.urlopen(url)
    return response.read().decode('utf-8')

def get_sequence(sequence_number: str) -> dict[int, int]:
    """Fetch and parse an OEIS sequence from cache or URL."""
    if not sequence_number.startswith('A'):
        raise ValueError("Sequence number must start with 'A'")
    
    cache_path = _get_cache_path(sequence_number)
    cache_valid = False
    
    if cache_path.exists():
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        cache_valid = cache_age < timedelta(days=7)
    
    if cache_valid:
        try:
            content = cache_path.read_text()
            return _parse_content(content, sequence_number)
        except (ValueError, IOError):
            # Cache invalid or corrupted, fallback to URL
            pass
    
    url = _construct_url(sequence_number)
    content = _fetch_from_url(url)
    
    # Ensure cache directory exists
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(content)
    
    return _parse_content(content, sequence_number)

# Updated OEIS_SEQUENCES with test_values
OEIS_SEQUENCES = {
    'A001415': { 'type': 'rect', 'param': 2, 'test_values': [0, 1, 3] },
    'A001416': { 'type': 'rect', 'param': 3, 'test_values': [0, 1, 3] },
    'A001417': { 'type': 'repeated', 'param': 2, 'test_values': [0, 1, 3] },
    'A001418': { 'type': 'square', 'param': None, 'test_values': [1, 3] },
    'A195646': { 'type': 'repeated', 'param': 3, 'test_values': [0, 1, 2] }
    # 'A007822': count_symmetric  # ignore for now
}

def calculate_sequence(sequence_id: str, n: int) -> int:
    """Calculate a(n) of an OEIS sequence."""
    if sequence_id not in OEIS_SEQUENCES:
        raise KeyError(f"Sequence {sequence_id} is not supported")
    sequence_info = OEIS_SEQUENCES[sequence_id]
    seq_type = sequence_info['type']
    param = sequence_info['param']
    # if sequence_id == 'A007822':
    #     # ...existing code...
    #     pass  # ignore for now

    if seq_type == 'rect':
        dims = [param, n]
    elif seq_type == 'repeated':
        dims = [param] * n
    elif seq_type == 'square':
        dims = [n, n]
    else:
        raise ValueError(f"Unknown sequence type '{seq_type}'")
    return count_foldings(dims) if n > 0 else 1