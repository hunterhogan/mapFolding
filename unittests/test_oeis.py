import unittest
from mapFolding import settingsOEISsequences, oeisSequence_aOFn, getOEISsequence

class TestCalculateSequence(unittest.TestCase):
    def test_calculate_sequence(self):
        for seq_id, info in settingsOEISsequences.items():
            test_values = info.get('testValuesValidation', [])
            expected_sequence = getOEISsequence(seq_id)
            for n in test_values:
                with self.subTest(sequence_id=seq_id, n=n):
                    result = oeisSequence_aOFn(seq_id, n)
                    self.assertIsInstance(result, int)
                    expected = expected_sequence.get(n)
                    if expected is not None:
                        self.assertEqual(result, expected)
                    else:
                        self.fail(f"Expected value for n={n} not found in sequence {seq_id}")

if __name__ == '__main__':
    unittest.main()