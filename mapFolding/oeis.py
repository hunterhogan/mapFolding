import pathlib
import urllib.request
from datetime import datetime, timedelta

from mapFolding import count_foldings

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

def get_sequence(sequence_number: str) -> dict[int, int]:
    """Fetch and parse an OEIS sequence from cache or URL."""
    cache_path = pathlib.Path(__file__).parent / ".cache" / f"{sequence_number}.txt"
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
    
    # Construct URL
    url = f"https://oeis.org/{sequence_number}/b{sequence_number[1:]}.txt"
    # Fetch content from URL
    response = urllib.request.urlopen(url)
    content = response.read().decode('utf-8')
    
    # Ensure cache directory exists
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(content)
    
    return _parse_content(content, sequence_number)

OEIS_SEQUENCES = {
    'A001415': { 'type': 'rect', 'param': 2, 'test_values': [0, 1, 3] },
    'A001416': { 'type': 'rect', 'param': 3, 'test_values': [0, 1, 3] },
    'A001417': { 'type': 'repeated', 'param': 2, 'test_values': [0, 1, 3] },
    'A001418': { 'type': 'square', 'param': None, 'test_values': [1, 3] },
    'A195646': { 'type': 'repeated', 'param': 3, 'test_values': [0, 1, 2] }
}

def calculate_sequence(sequence_id: str, n: int) -> int:
    """Calculate a(n) of an OEIS sequence."""
    if sequence_id not in OEIS_SEQUENCES:
        raise KeyError(f"Sequence {sequence_id} is not supported")
    sequence_info = OEIS_SEQUENCES[sequence_id]
    seq_type = sequence_info['type']
    param = sequence_info['param']

    if seq_type == 'rect':
        dims = [param, n]
    elif seq_type == 'repeated':
        dims = [param] * n
    elif seq_type == 'square':
        dims = [n, n]
    else:
        raise ValueError(f"Unknown sequence type '{seq_type}'")
    return count_foldings(dims) if n > 0 else 1