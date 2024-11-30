import pathlib
import multiprocessing
import random
import unittest
import urllib.request

from mapFolding import computeDistributedTask, computeSeries, computeSeriesConcurrently, sumDistributedTasks, pathTasksToParameters


class MapFoldingTestSuite(unittest.TestCase):
    """Base class for map folding tests with shared functionality"""
    
    @classmethod
    def setUpClass(cls):
        cls.oeisTestConfigurations = {
            'A001415': { 'series': '2', 'testValues': [1, random.randint(2, 9)], 'url': 'https://oeis.org/A001415/b001415.txt', 
                        'pathTest': '/apps/mapFolding/unittests/2/7/13/True',
            'expectedIndexValues': [0, 3794, 2590, 3136, 2156, 3668, 1890, 5180, 3108, 6692, 2436, 17990, 8148]
            },
            'A001416': { 'series': '3', 'testValues': [1, random.randint(2, 5)], 'url': 'https://oeis.org/A001416/b001416.txt', 
                        'pathTest': '/apps/mapFolding/unittests/3/5/11/True',
            'expectedIndexValues': [0, 5475, 18735, 11940, 5775, 7770, 6750, 0, 54360, 24300, 66135,]
            },
            'A001417': { 'series': '2 X 2', 'testValues': [1, random.randint(2, 3)], 'url': 'https://oeis.org/A001417/b001417.txt', 'pathTest': '/apps/mapFolding/unittests/2 X 2/4/15/True',
            'expectedIndexValues': [0, 0, 0, 0, 0, 0, 0, 1152, 384, 0, 0, 1152, 384, 1152, 384,]
            },
            'A001418': { 'series': 'n', 'testValues': [1, random.randint(2, 3)], 'url': 'https://oeis.org/A001418/b001418.txt', 'pathTest': '/apps/mapFolding/unittests/n/4/13/True',
            'expectedIndexValues': [0, 8193025, 5863425, 17882450, 9327900, 10264800, 4942650, 0, 35828000, 14250750, 10624150, 23992375, 44917075,]
            }
        }
        
        # Load all sequences
        cls.sequencesOEIS = {
            valuesConfirmed: cls._getValuesConfirmed(configuration['url'])
            for valuesConfirmed, configuration in cls.oeisTestConfigurations.items()
        }

        COUNTcpu = multiprocessing.cpu_count()
        cls.testValuesCPUlimit = [
            None, True, False, 0, 1, 0.5, -1, -0.5,
            random.randint(2, COUNTcpu - 1),
            -random.randint(2, COUNTcpu - 1),
            random.uniform(0.01, 1),
            -random.uniform(0.01, 1)
        ]

    @staticmethod
    def _getValuesConfirmed(url: str) -> dict[int, int]:
        valuesConfirmed = {}
        with urllib.request.urlopen(url) as httpRead:
            for line in httpRead:
                if line.startswith(b'#'):
                    continue
                n_as_str, aOFn_as_str = line.decode().strip().split()
                n = int(n_as_str)
                valuesConfirmed[n] = int(aOFn_as_str)
        return valuesConfirmed

    def validateComputation(self, OEISid: str, n: int, result: int) -> None:
        expected = self.sequencesOEIS[OEISid].get(n)
        self.assertEqual(
            result, 
            expected, 
            f"{OEISid}, n={n}: expected {expected} but got {result}"
        )

class OEISValidations(MapFoldingTestSuite):
    """OEIS sequence validation tests"""

    def runOEISvalidation(self, OEISid: str, concurrent: bool = False) -> None:
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

    def test_CPUlimit_A001415(self):
        configuration = self.oeisTestConfigurations['A001415']
        n = configuration['testValues'][-1]  # Use a small test value for speed
        
        for CPUlimit in self.testValuesCPUlimit:
            with self.subTest(cpu_limit=CPUlimit):
                result = computeSeriesConcurrently(configuration['series'], n, CPUlimit=CPUlimit)
                self.validateComputation('A001415', n, result)

    def test_CPUlimit_allSeries(self):
        # Test one CPU limit across all series
        CPUlimit = random.choice(self.testValuesCPUlimit)
        for OEISid, configuration in self.oeisTestConfigurations.items():
            n = configuration['testValues'][-1]  # Use a small test value for speed
            with self.subTest(series=OEISid):
                result = computeSeriesConcurrently(configuration['series'], n, CPUlimit=CPUlimit)
                self.validateComputation(OEISid, n, result)

class TestDistributedTasks(MapFoldingTestSuite):
    """Tests for computeDistributedTask function"""

    def test_computeDistributedTask(self):
        for OEISid, configuration in self.oeisTestConfigurations.items():
            pathTasks = pathlib.Path(configuration['pathTest'])
            for pathFilename in pathTasks.glob('*.computationIndex'):
                pathFilename.unlink()
            CPUlimit = 2  # Test two random indices per testPath
            computeDistributedTask(pathTasks, CPUlimit=CPUlimit)
            for indexCompleted in pathTasks.glob('*'):
                with self.subTest(OEISid=OEISid):
                    result = int(indexCompleted.read_text())
                    index = int(indexCompleted.stem)
                    expected = configuration['expectedIndexValues'][index]
                    self.assertEqual(result, expected, f"Failed at index {index}: expected {expected} but got {result}")
                    
    def test_sumDistributedTasks(self):
        for OEISid, configuration in self.oeisTestConfigurations.items():
            pathTasks = pathlib.Path(configuration['pathTest'])
            # Delete any existing computationIndex files
            for pathFilename in pathTasks.glob('*.computationIndex'):
                pathFilename.unlink()
            # Compute all distributed tasks
            with self.subTest(OEISid=OEISid):
                while computeDistributedTask(pathTasks, CPUlimit=random.choice(self.testValuesCPUlimit)) > 0:
                    pass
                result = sumDistributedTasks(pathTasks)
                if not result:
                    result = -1
                series, X_n, computationDivisions, normalFoldings = pathTasksToParameters(pathTasks)
                self.validateComputation(OEISid, X_n, result)
            # Clean up the computationIndex files after test
            for pathFilename in pathTasks.glob('*.computationIndex'):
                pathFilename.unlink()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    unittest.main()