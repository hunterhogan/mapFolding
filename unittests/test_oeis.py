import unittest
from mapFolding import OEIS_SEQUENCES, calculate_sequence, get_sequence

class TestCalculateSequence(unittest.TestCase):
    def test_calculate_sequence(self):
        for seq_id, info in OEIS_SEQUENCES.items():
            test_values = info.get('test_values', [])
            expected_sequence = get_sequence(seq_id)
            for n in test_values:
                with self.subTest(sequence_id=seq_id, n=n):
                    result = calculate_sequence(seq_id, n)
                    self.assertIsInstance(result, int)
                    expected = expected_sequence.get(n)
                    if expected is not None:
                        self.assertEqual(result, expected)
                    else:
                        self.fail(f"Expected value for n={n} not found in sequence {seq_id}")

if __name__ == '__main__':
    unittest.main()