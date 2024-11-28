import multiprocessing
import random
import unittest
import urllib.request

from mapFolding import computeSeries, computeSeriesConcurrently


class MapFoldingTestSuite(unittest.TestCase):
    """Base class for map folding tests with shared functionality"""
    
    @classmethod
    def setUpClass(cls):
        cls.oeisTestConfigurations = {
            'A001415': {'series': '2', 'testValues': [0,1, random.randint(2, 9)],'url': 'https://oeis.org/A001415/b001415.txt'},
            'A001416': {'series': '3', 'testValues': [0,1, random.randint(2, 5)], 'url': 'https://oeis.org/A001416/b001416.txt'},
            'A001417': {'series':'2 X 2','testValues':[0,1, random.randint(2, 3)], 'url': 'https://oeis.org/A001417/b001417.txt'},
            'A001418': {'series': 'n', 'testValues': [0,1, random.randint(2, 3)], 'url': 'https://oeis.org/A001418/b001418.txt'}
        }
        
        # Load all sequences
        cls.sequencesOEIS = {
            valuesConfirmed: cls._getValuesConfirmed(configuration['url'])
            for valuesConfirmed, configuration in cls.oeisTestConfigurations.items()
        }

    @staticmethod
    def _getValuesConfirmed(url):
        valuesConfirmed = {}
        with urllib.request.urlopen(url) as httpRead:
            for line in httpRead:
                if line.startswith(b'#'):
                    continue
                n_as_str, aOFn_as_str = line.decode().strip().split()
                n = int(n_as_str)
                valuesConfirmed[n] = int(aOFn_as_str)
        return valuesConfirmed

    def validateComputation(self, OEISid, n, result):
        expected = self.sequencesOEIS[OEISid].get(n)
        self.assertEqual(
            result, 
            expected, 
            f"{OEISid} failed at n={n}: expected {expected} but got {result}"
        )

class OEISValidations(MapFoldingTestSuite):
    """OEIS sequence validation tests"""

    def runOEISvalidation(self, OEISid, concurrent=False):
        configuration = self.oeisTestConfigurations[OEISid]
        for n in configuration['testValues']:
            if concurrent:
                result = computeSeriesConcurrently(configuration['series'], n)
            else:
                result = computeSeries(configuration['series'], n)
            self.validateComputation(OEISid, n, result)

    def test_A001415(self):
        self.runOEISvalidation('A001415')

    def test_A001416(self):
        self.runOEISvalidation('A001416')

    def test_A001417(self):
        self.runOEISvalidation('A001417')

    def test_A001418(self):
        self.runOEISvalidation('A001418')

    def test_A001415concurrently(self):
        self.runOEISvalidation('A001415', concurrent=True)

    def test_A001416concurrently(self):
        self.runOEISvalidation('A001416', concurrent=True)

    def test_A001417concurrently(self):
        self.runOEISvalidation('A001417', concurrent=True)

    def test_A001418concurrently(self):
        self.runOEISvalidation('A001418', concurrent=True)

class TestCPUlimitParameter(MapFoldingTestSuite):
    """Tests for computeSeriesConcurrently with various CPU limits"""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        COUNTcpu = multiprocessing.cpu_count()
        cls.testValues = [
            None, True, False, 0, 1, 0.5, -1, -0.5,
            random.randint(2, COUNTcpu - 1),
            -random.randint(2, COUNTcpu - 1),
            random.uniform(0.01, 1),
            -random.uniform(0.01, 1)
        ]

    def test_CPUlimit_A001415(self):
        config = self.oeisTestConfigurations['A001415']
        n = config['testValues'][-1]  # Use a small test value for speed
        
        for CPUlimit in self.testValues:
            with self.subTest(cpu_limit=CPUlimit):
                result = computeSeriesConcurrently(config['series'], n, CPUlimit=CPUlimit)
                self.validateComputation('A001415', n, result)

    def test_CPUlimit_all_series(self):
        # Test one CPU limit across all series
        CPUlimit = random.choice(self.testValues)
        for OEISid, config in self.oeisTestConfigurations.items():
            n = config['testValues'][-1]  # Use a small test value for speed
            with self.subTest(series=OEISid):
                result = computeSeriesConcurrently(config['series'], n, CPUlimit=CPUlimit)
                self.validateComputation(OEISid, n, result)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    unittest.main()