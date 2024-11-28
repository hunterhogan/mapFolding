import unittest
import urllib.request
from mapFolding import computeSeries, computeSeriesConcurrently

class TestMapFolding(unittest.TestCase):
    """
    OEIS sequences:
        A001415 Number of ways of folding a 2 X n strip of stamps.
        A001416 Number of ways of folding a 3 X n strip of stamps.
        A001417 Number of ways of folding a 2 X 2 X ... X 2 n-dimensional map.
        A001418 Number of ways of folding an n X n sheet of stamps.
    """
    def setUp(self):
        # Define test configurations
        self.test_cases = {
            'A001415': {'series': '2', 'testValues': list(range(0, 10)), 'url': 'https://oeis.org/A001415/b001415.txt'},
            'A001416': {'series': '3', 'testValues': list(range(0, 6)), 'url': 'https://oeis.org/A001416/b001416.txt'},
            'A001417': {'series': '2 X 2', 'testValues': list(range(0, 3)), 'url': 'https://oeis.org/A001417/b001417.txt'},
            'A001418': {'series': 'n', 'testValues': list(range(0, 4)), 'url': 'https://oeis.org/A001418/b001418.txt'}
        }
        
        # Load all sequences
        self.sequences = {
            sequence: self.load_sequence(config['url'])
            for sequence, config in self.test_cases.items()
        }

    def load_sequence(self, url):
        sequence = {}
        with urllib.request.urlopen(url) as httpRead:
            for line in httpRead:
                if line.startswith(b'#'):
                    continue  # Skip comments
                n_str, a_n_str = line.decode().strip().split()
                n = int(n_str)
                sequence[n] = int(a_n_str)
        return sequence

    def run_sequence_test(self, sequence_id):
        """Generic test runner for any OEIS sequence"""
        config = self.test_cases[sequence_id]
        for n in config['testValues']:
            result = computeSeries(config['series'], n)
            expected = self.sequences[sequence_id].get(n)
            self.assertEqual(
                result, 
                expected, 
                f"{sequence_id} failed at n={n}: expected {expected}, got {result}"
            )

    def test_A001415(self):
        self.run_sequence_test('A001415')

    def test_A001416(self):
        self.run_sequence_test('A001416')

    def test_A001417(self):
        self.run_sequence_test('A001417')

    def test_A001418(self):
        self.run_sequence_test('A001418')

    def test_concurrent_A001415(self):
        """Test concurrent computation with different CPU configurations"""
        config = self.test_cases['A001415']
        n = 5  # Using a moderate size for concurrent testing
        expected = self.sequences['A001415'].get(n)

        # Test with default CPU count
        result = computeSeriesConcurrently(config['series'], n)
        self.assertEqual(result, expected)

        # Test with specific CPU limits
        cpu_configs = [1, 2, 0.5, -1, True, False]
        for cpu_limit in cpu_configs:
            result = computeSeriesConcurrently(config['series'], n, CPUlimit=cpu_limit)
            self.assertEqual(
                result, 
                expected, 
                f"Concurrent computation failed with CPU limit {cpu_limit}"
            )

    def test_concurrent_matches_serial(self):
        """Test that concurrent results match serial computation"""
        test_cases = [
            ('2', 4),
            ('3', 3),
            ('2 X 2', 2),
            ('n', 2)
        ]

        for series, n in test_cases:
            serial_result = computeSeries(series, n)
            concurrent_result = computeSeriesConcurrently(series, n)
            self.assertEqual(
                concurrent_result,
                serial_result,
                f"Concurrent computation mismatch for series={series}, n={n}"
            )

if __name__ == "__main__":
    unittest.main()