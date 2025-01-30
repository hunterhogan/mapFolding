from typing import Any, Callable, Sequence, Type, Union
import pytest

def uniformTestMessage(expected: Any, actual: Any, functionName: str, *arguments: Any) -> str:
    """Format assertion message for any test comparison."""
    return (f"\nTesting: `{functionName}({', '.join(str(parameter) for parameter in arguments)})`\n"
            f"Expected: {expected}\n"
            f"Got: {actual}")

def standardizedEqualTo(expected: Any, functionTarget: Callable, *arguments: Any) -> None:
    """Template for tests expecting an error."""
    if type(expected) is Type[Exception]:
        messageExpected = expected.__name__
    else:
        messageExpected = expected

    try:
        messageActual = actual = functionTarget(*arguments)
    except Exception as actualError:
        messageActual = type(actualError).__name__
        actual = type(actualError)

    assert actual == expected, uniformTestMessage(messageExpected, messageActual, functionTarget.__name__, *arguments)

def standardizedSystemExit(expected: Union[str, int, Sequence[int]], functionTarget: Callable, *arguments: Any) -> None:
    """Template for tests expecting SystemExit.

    Parameters
        expected: Exit code expectation:
            - "error": any non-zero exit code
            - "nonError": specifically zero exit code
            - int: exact exit code match
            - Sequence[int]: exit code must be one of these values
        functionTarget: The function to test
        arguments: Arguments to pass to the function
    """
    with pytest.raises(SystemExit) as exitInfo:
        functionTarget(*arguments)

    exitCode = exitInfo.value.code

    if expected == "error":
        assert exitCode != 0, \
            f"Expected error exit (non-zero) but got code {exitCode}"
    elif expected == "nonError":
        assert exitCode == 0, \
            f"Expected non-error exit (0) but got code {exitCode}"
    elif isinstance(expected, (list, tuple)):
        assert exitCode in expected, \
            f"Expected exit code to be one of {expected} but got {exitCode}"
    else:
        assert exitCode == expected, \
            f"Expected exit code {expected} but got {exitCode}"
