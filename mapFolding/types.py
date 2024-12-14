from typing import Callable, Dict, List, Literal

OEISsequenceID = Literal['A001415', 'A001416', 'A001417', 'A195646', 'A001418']

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

class SettingsOEISsequence(TypedDict):
    description: str
    dimensions: Callable[[int], List[int]]
    benchmarkValues: List[int]
    testValuesValidation: List[int]
    valuesKnown: Dict[int, int]
    valueUnknown: int
