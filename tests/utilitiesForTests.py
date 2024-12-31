from typing import Any, Callable, Dict, Tuple, Type, TypeVar, Optional, Union, Sequence
import pytest
from mapFolding.oeis import settingsOEISsequences

def generateDictionaryDimensionsFoldingsTotal() -> Dict[Tuple[int,...], int]:
    """Returns a dictionary mapping dimension tuples to their known folding totals."""
    dimensionsFoldingsTotalLookup = {}
    
    for settings in settingsOEISsequences.values():
        sequence = settings['valuesKnown']
        
        for n, foldingsTotal in sequence.items():
            dimensions = settings['getDimensions'](n)
            dimensions.sort()
            dimensionsFoldingsTotalLookup[tuple(dimensions)] = foldingsTotal
    
    return dimensionsFoldingsTotalLookup

# Template Types
ReturnType = TypeVar('ReturnType')
ErrorTypes = Union[Type[Exception], Tuple[Type[Exception], ...]]

def formatTestMessage(
    expected: Any, actual: Any, 
    functionName: str, 
    *arguments: Any) -> str:
    """Format assertion message for any test comparison."""
    return (f"\nTesting: `{functionName}({', '.join(str(parameter) for parameter in arguments)})`\n"
            f"Expected: {expected}\n"
            f"Got: {actual}")

def compareValues(expected: Any, functionTarget: Callable, *arguments: Any) -> None:
    """Template for tests comparing function output to expected value."""
    actual = functionTarget(*arguments)
    assert actual == expected, formatTestMessage(functionTarget.__name__, expected, actual, *arguments)

def expectError(expected: Type[Exception], functionTarget: Callable, *arguments: Any) -> None:
    """Template for tests expecting an error."""
    try:
        actualName = actualObject = functionTarget(*arguments)
    except Exception as actualError:
        actualName = type(actualError).__name__
        actualObject = actualError

    assert isinstance(actualObject, expected), \
            formatTestMessage(expected.__name__, actualName, functionTarget.__name__, *arguments)
