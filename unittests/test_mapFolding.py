import pathlib
import multiprocessing
import random
import unittest
import pickle
import time
import urllib.request
from typing import Dict

from mapFolding import computeDistributedTask, computeSeries, computeSeriesConcurrently, sumDistributedTasks, pathTasksToParameters


class MapFoldingTestSuite(unittest.TestCase):
    """Base class for map folding tests with shared functionality"""
    pathCache = pathlib.Path(__file__).parent / ".cache"
    CACHE_EXPIRY = 24 * 60 * 60

    @classmethod
    def setUpClass(superEgo):
        superEgo.pathCache.mkdir(parents=True, exist_ok=True)
        superEgo.oeisTestConfigurations = {
            'A001415': { 'series': 2, 'testValues': [0, 1, random.randint(2, 9)],
                        'pathTest': '/apps/mapFolding/unittests/2/7/13/True',
            'expectedIndexValues': [0, 3794, 2590, 3136, 2156, 3668, 1890, 5180, 3108, 6692, 2436, 17990, 8148]
            },
            'A001416': { 'series': 3, 'testValues': [0, 1, random.randint(2, 5)],
                        'pathTest': '/apps/mapFolding/unittests/3/5/11/True',
            'expectedIndexValues': [0, 5475, 18735, 11940, 5775, 7770, 6750, 0, 54360, 24300, 66135,]
            },
            'A001417': { 'series': '2 X 2', 'testValues': [0, 1, random.randint(2, 3)],
                        'pathTest': '/apps/mapFolding/unittests/2 X 2/4/15/True',
            'expectedIndexValues': [0, 0, 0, 0, 0, 0, 0, 1152, 384, 0, 0, 1152, 384, 1152, 384,]
            },
            'A195646': { 'series': '3 X 3', 'testValues': [0, 1, 2],
                        'pathTest': '',
            'expectedIndexValues': []
            },
            'A001418': { 'series': 'n', 'testValues': [1, random.randint(2, 3)],
                        'pathTest': '/apps/mapFolding/unittests/n/4/13/True',
            'expectedIndexValues': [0, 22064, 14688, 11136, 11600, 27712, 12496, 9664, 10448, 88432, 44480, 25072, 22816]
            },
            # 'A007822': { 'series': 'A007822', 'testValues': list(range(4,7)), 'pathTest': '',
            # 'expectedIndexValues': []
            # },
        }

        for OEISid in superEgo.oeisTestConfigurations:
            superEgo.oeisTestConfigurations[OEISid]['valuesConfirmed'] = superEgo._getOEISValues(OEISid)

        COUNTcpu = multiprocessing.cpu_count()
        superEgo.testValuesCPUlimit = [
                None, True, False, 0, 1, 0.5, -1, -0.5,
                random.randint(2, COUNTcpu - 1),
                -random.randint(2, COUNTcpu - 1),
                random.uniform(0.01, 1),
                -random.uniform(0.01, 1)
            ]

    @classmethod
    def _getOEISValues(superEgo, OEISid: str) -> Dict[int, int]:
        pathFilenameCache = superEgo.pathCache / f"{OEISid}.pkl"

        if pathFilenameCache.exists() and time.time() - pathFilenameCache.stat().st_mtime < superEgo.CACHE_EXPIRY:
            return pickle.loads(pathFilenameCache.read_bytes())

        url = f'https://oeis.org/{OEISid}/b{OEISid[1:]}.txt'
        valuesConfirmed = {}

        with urllib.request.urlopen(url) as readHTTP:
            for line in readHTTP:
                if line.startswith(b'#'):
                    continue
                n_as_str, aOFn_as_str = line.decode().strip().split()
                n = int(n_as_str)
                valuesConfirmed[n] = int(aOFn_as_str)

        pathFilenameCache.write_bytes(pickle.dumps(valuesConfirmed))
        return valuesConfirmed

    def validateComputation(self, OEISid: str, n: int, result: int) -> None:
        expected = self.oeisTestConfigurations[OEISid]['valuesConfirmed'][n]
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

class TestCPUlimitParameter(MapFoldingTestSuite):
    """Tests for computeSeriesConcurrently with various CPU limits"""

    @classmethod
    def setUpClass(superEgo):
        super().setUpClass()

    def test_CPUlimit(self):
        for CPUlimit in self.testValuesCPUlimit:
            OEISid = random.choice(list(self.oeisTestConfigurations.keys()))
            configuration = self.oeisTestConfigurations[OEISid]
            n = configuration['testValues'][-1]
            with self.subTest(cpu_limit=CPUlimit, series=OEISid, n=n):
                result = computeSeriesConcurrently(configuration['series'], n, CPUlimit=CPUlimit)
                self.validateComputation(OEISid, n, result)

# class TestDistributedTasks(MapFoldingTestSuite):
#     """Tests for computeDistributedTask function"""

#     def test_computeDistributedTask(self):
#         for OEISid, configuration in self.oeisTestConfigurations.items():
#             pathTasks = pathlib.Path(configuration['pathTest'])
#             for pathFilename in pathTasks.glob('*.computationIndex'):
#                 pathFilename.unlink()
#             CPUlimit = 2  # Test two random indices per testPath
#             computeDistributedTask(pathTasks, CPUlimit=CPUlimit)
#             for indexCompleted in pathTasks.glob('*.computationIndex'):
#                 with self.subTest(OEISid=OEISid):
#                     result = int(indexCompleted.read_text())
#                     index = int(indexCompleted.stem)
#                     expected = configuration['expectedIndexValues'][index]
#                     self.assertEqual(result, expected, f"Failed at index {index}: expected {expected} but got {result}")

#     def test_sumDistributedTasks(self):
#         OEISid = random.choice(list(self.oeisTestConfigurations.keys()))
#         configuration = self.oeisTestConfigurations[OEISid]
#         pathTasks = pathlib.Path(configuration['pathTest'])
#         # Delete any existing computationIndex files
#         for pathFilename in pathTasks.glob('*.computationIndex'):
#             pathFilename.unlink()
#         # Choose CPU limit once
#         CPUlimit = random.choice(self.testValuesCPUlimit)
#         # Compute all distributed tasks
#         with self.subTest(OEISid=OEISid, CPUlimit=CPUlimit):
#             while computeDistributedTask(pathTasks, CPUlimit=CPUlimit) > 0:
#                 pass
#             result = sumDistributedTasks(pathTasks)
#             if not result:
#                 result = -1
#             series, X_n, computationDivisions, normalFoldings = pathTasksToParameters(pathTasks)
#             self.validateComputation(OEISid, X_n, result)
#         # Clean up the computationIndex files after test
#         for pathFilename in pathTasks.glob('*.computationIndex'):
#             pathFilename.unlink()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    unittest.main()