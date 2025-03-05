import pathlib
from collections.abc import Callable as Callable
from mapFolding import countFolds as countFolds
from mapFolding.theSSOT import pathPackage as pathPackage
from pathlib import Path
from typing import Any, Final, TypedDict

cacheDays: int
pathCache: Path

class SettingsOEIS(TypedDict):
    description: str
    getMapShape: Callable[[int], tuple[int, ...]]
    offset: int
    valuesBenchmark: list[int]
    valuesKnown: dict[int, int]
    valuesTestParallelization: list[int]
    valuesTestValidation: list[int]
    valueUnknown: int

class SettingsOEIShardcodedValues(TypedDict):
    getMapShape: Callable[[int], tuple[int, ...]]
    valuesBenchmark: list[int]
    valuesTestParallelization: list[int]
    valuesTestValidation: list[int]

settingsOEIShardcodedValues: dict[str, SettingsOEIShardcodedValues]
oeisIDsImplemented: Final[list[str]]

def validateOEISid(oeisIDcandidate: str) -> str: ...
def getFilenameOEISbFile(oeisID: str) -> str: ...
def _parseBFileOEIS(OEISbFile: str, oeisID: str) -> dict[int, int]: ...
def getOEISofficial(pathFilenameCache: pathlib.Path, url: str) -> None | str: ...
def getOEISidValues(oeisID: str) -> dict[int, int]: ...
def getOEISidInformation(oeisID: str) -> tuple[str, int]: ...
def makeSettingsOEIS() -> dict[str, SettingsOEIS]: ...

settingsOEIS: dict[str, SettingsOEIS]

def _formatHelpText() -> str: ...
def _formatOEISsequenceInfo() -> str: ...
def oeisIDfor_n(oeisID: str, n: int | Any) -> int: ...
def OEIS_for_n() -> None: ...
def clearOEIScache() -> None: ...
def getOEISids() -> None: ...
