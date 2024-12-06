import random
import pathlib
from typing import Callable, Dict, List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypedDict
else:
    TypedDict = dict

class SettingsOEISsequence(TypedDict):
    dimensions: Callable[[int], List[int]]
    testValuesValidation: List[int]
    testValuesSpeed: List[int]
    description: str

try:
    pathCache = pathlib.Path(__file__).parent / ".cache"
except NameError:
    pathCache = pathlib.Path.home() / ".mapFoldingCache"

settingsOEISsequences: Dict[str, SettingsOEISsequence] = {
    'A001415': {
        'description': 'Number of ways of folding a 2 X n strip of stamps.',
        'dimensions': lambda n: [2, n],
        'testValuesSpeed': [11],
        'testValuesValidation': [*range(0, 10)],
        # 'testValuesValidation': [0, 1, random.randint(2, 9)],
                },
    'A001416': {
        'description': 'Number of ways of folding a 3 X n strip of stamps.',
        'dimensions': lambda n: [3, n],
        'testValuesSpeed': [8],
        'testValuesValidation': [0, 1, random.randint(2, 6)],
    },
    'A001417': {
        'description': 'Number of ways of folding a 2 X 2 X ... X 2 n-dimensional map.',
        'dimensions': lambda n: [2] * n,
        'testValuesSpeed': [5],
        'testValuesValidation': [0, 1, random.randint(2, 4)],
    },
    'A195646': {
        'description': 'Number of ways of folding a 3 X 3 X ... X 3 n-dimensional map.',
        'dimensions': lambda n: [3] * n,
        'testValuesSpeed': [3],
        'testValuesValidation': [0, 1, 2],
    },
    'A001418': {
        'description': 'Number of ways of folding an n X n sheet of stamps.',
        'dimensions': lambda n: [n, n],
        'testValuesSpeed': [5],
        'testValuesValidation': [*range(1, 6)],
        # 'testValuesValidation': [1, random.randint(2, 4)],
    },
}
