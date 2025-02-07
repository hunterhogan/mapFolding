from collections import defaultdict
from numpy import integer
from types import ModuleType
from typing import Any, Callable, Dict, Final, Optional, Tuple, Type, TypedDict, cast
import enum
import numba
import numpy
import numpy.typing
import pathlib
import sys

"""I have hobbled together:
TypedDict, Enum, defaultdict, and lookup dictionaries to make DIY immutability and delayed realization/instantiation.
Nevertheless, I am both confident that all of these processes will be replaced and completely ignorant of what will replace them."""

"""Technical concepts I am likely using and likely want to use more effectively:
- Configuration Registry
- Write-Once, Read-Many (WORM) / Immutable Initialization
- Lazy Initialization
- Separation of Concerns: in the sense that configuration is separated from business logic

Furthermore, I want to more clearly divorce the concept of a single _source_ of (a) truth from
the _authority_ of that truth. The analogy to a registry of ownership is still apt: the registry
is, at most, a single (or centralized) source of truth, but it is merely the place to register/record
the truth determined by some other authority.

And, I almost certainly want to change the semiotics from "authority" (of truth) to "power" (to create a truth).
Here, "power" is a direct analogy to https://hunterthinks.com/opinion/a-hohfeldian-primer.
"""

myPackageNameIs = "mapFolding"

moduleOfSyntheticModules = "syntheticModules"

def getPathPackage() -> pathlib.Path:
    import importlib, inspect
    pathPackage = pathlib.Path(inspect.getfile(importlib.import_module(myPackageNameIs)))
    if pathPackage.is_file():
        pathPackage = pathPackage.parent
    return pathPackage

def getPathJobDEFAULT() -> pathlib.Path:
    if 'google.colab' in sys.modules:
        pathJobDEFAULT = pathlib.Path("/content/drive/MyDrive") / "jobs"
    else:
        pathJobDEFAULT = getPathPackage() / "jobs"
    return pathJobDEFAULT

def getPathSyntheticModules() -> pathlib.Path:
    pathSyntheticModules = getPathPackage() / moduleOfSyntheticModules
    return pathSyntheticModules

def getAlgorithmSource() -> ModuleType:
    from mapFolding import theDao
    return theDao

def getAlgorithmCallable() -> Callable[..., None]:
    algorithmSource = getAlgorithmSource()
    return cast(Callable[..., None], algorithmSource.doTheNeedful)

def getDispatcherCallable() -> Callable[..., None]:
    from mapFolding.syntheticModules import numba_doTheNeedful
    return cast(Callable[..., None], numba_doTheNeedful.doTheNeedful)

# NOTE I want this _concept_ to be well implemented and usable everywhere: Python, Numba, Jax, CUDA, idc
class computationState(TypedDict):
    connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]]
    foldGroups: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
    gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
    mapShape: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
    my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
    track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]

@enum.verify(enum.CONTINUOUS, enum.UNIQUE) if sys.version_info >= (3, 11) else lambda x: x
class EnumIndices(enum.IntEnum):
    """Base class for index enums."""
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[Any]) -> int:
        """0-indexed."""
        return count

    def __index__(self) -> int:
        """Adapt enum to the ultra-rare event of indexing a NumPy 'ndarray', which is not the
        same as `array.array`. See NumPy.org; I think it will be very popular someday."""
        return self.value

class indexMy(EnumIndices):
    """Indices for scalar values."""
    dimensionsTotal = enum.auto()
    dimensionsUnconstrained = enum.auto()
    gap1ndex = enum.auto()
    gap1ndexCeiling = enum.auto()
    indexDimension = enum.auto()
    indexLeaf = enum.auto()
    indexMiniGap = enum.auto()
    leaf1ndex = enum.auto()
    leafConnectee = enum.auto()
    taskDivisions = enum.auto()
    taskIndex = enum.auto()

class indexTrack(EnumIndices):
    """Indices for state tracking array."""
    leafAbove = enum.auto()
    leafBelow = enum.auto()
    countDimensionsGapped = enum.auto()
    gapRangeStart = enum.auto()

class ParametersNumba(TypedDict):
    _nrt: bool
    boundscheck: bool
    cache: bool
    error_model: str
    fastmath: bool
    forceinline: bool
    inline: str
    looplift: bool
    no_cfunc_wrapper: bool
    no_cpython_wrapper: bool
    nopython: bool
    parallel: bool

parametersNumbaDEFAULT: Final[ParametersNumba] = {
    '_nrt': True,
    'boundscheck': False,
    'cache': True,
    'error_model': 'numpy',
    'fastmath': True,
    'forceinline': False,
    'inline': 'never',
    'looplift': False,
    'no_cfunc_wrapper': True,
    'no_cpython_wrapper': True,
    'nopython': True,
    'parallel': False,
}

"delay realization/instantiation until a concrete value is desired"
"moment of truth: when the value is needed, not when the value is defined"

"""What is a (not too complicated, integer) datatype?
    - ecosystem/module
        - must apathy|value|list of values
        - mustn't apathy|value|list of values
    - bit width
        - bits maximum apathy|value
        - bits minimum apathy|value
        - magnitude maximum apathy|value
        - ?magnitude minimum apathy|value
    - signedness apathy|non-negative|non-positive|both
    """

_datatypeDefault: Final[Dict[str, str]] = {
    'elephino': 'uint8',
    'foldsTotal': 'int64',
    'leavesTotal': 'uint8',
}
_datatypeModule = ''
_datatypeModuleDEFAULT: Final[str] = 'numpy'

_datatype: Dict[str, str] = defaultdict(str)

def reportDatatypeLimit(identifier: str, datatype: str, sourGrapes: Optional[bool] = False) -> str:
    global _datatype
    if not _datatype[identifier]:
        _datatype[identifier] = datatype
    elif _datatype[identifier] == datatype:
        pass
    elif sourGrapes:
        raise Exception(f"Datatype is '{_datatype[identifier]}' not '{datatype}', so you can take your ball and go home.")
    return _datatype[identifier]

def setDatatypeModule(datatypeModule: str, sourGrapes: Optional[bool] = False) -> str:
    global _datatypeModule
    if not _datatypeModule:
        _datatypeModule = datatypeModule
    elif _datatypeModule == datatypeModule:
        pass
    elif sourGrapes:
        raise Exception(f"Datatype module is '{_datatypeModule}' not '{datatypeModule}', so you can take your ball and go home.")
    return _datatypeModule

def setDatatypeElephino(datatype: str, sourGrapes: Optional[bool] = False) -> str:
    return reportDatatypeLimit('elephino', datatype, sourGrapes)

def setDatatypeFoldsTotal(datatype: str, sourGrapes: Optional[bool] = False) -> str:
    return reportDatatypeLimit('foldsTotal', datatype, sourGrapes)

def setDatatypeLeavesTotal(datatype: str, sourGrapes: Optional[bool] = False) -> str:
    return reportDatatypeLimit('leavesTotal', datatype, sourGrapes)

def _get_datatype(identifier: str) -> str:
    global _datatype
    if not _datatype[identifier]:
        if identifier in indexMy._member_names_:
            _datatype[identifier] = _datatypeDefault.get(identifier) or _get_datatype('elephino')
        elif identifier in indexTrack._member_names_:
            _datatype[identifier] = _datatypeDefault.get(identifier) or _get_datatype('elephino')
        else:
            _datatype[identifier] = _datatypeDefault.get(identifier) or _get_datatype('foldsTotal')
    return _datatype[identifier]

def _getDatatypeModule() -> str:
    global _datatypeModule
    if not _datatypeModule:
        _datatypeModule = _datatypeModuleDEFAULT
    return _datatypeModule

def setInStone(identifier: str) -> Type[Any]:
    datatypeModule = _getDatatypeModule()
    datatypeStr = _get_datatype(identifier)
    return cast(Type[Any], getattr(eval(datatypeModule), datatypeStr))

def hackSSOTdtype(identifier: str) -> Type[Any]:
    _hackSSOTdtype={
    'connectionGraph': 'dtypeLeavesTotal',
    'dtypeElephino': 'dtypeElephino',
    'dtypeFoldsTotal': 'dtypeFoldsTotal',
    'dtypeLeavesTotal': 'dtypeLeavesTotal',
    'foldGroups': 'dtypeFoldsTotal',
    'gapsWhere': 'dtypeLeavesTotal',
    'mapShape': 'dtypeLeavesTotal',
    'my': 'dtypeElephino',
    'track': 'dtypeElephino',
    }
    RubeGoldBerg = _hackSSOTdtype[identifier]
    if RubeGoldBerg == 'dtypeElephino':
        return setInStone('elephino')
    elif RubeGoldBerg == 'dtypeFoldsTotal':
        return setInStone('foldsTotal')
    elif RubeGoldBerg == 'dtypeLeavesTotal':
        return setInStone('leavesTotal')
    raise Exception("Dude, you forgot to set a value in `hackSSOTdtype`.")

def hackSSOTdatatype(identifier: str) -> str:
    _hackSSOTdatatype={
    'connectionGraph': 'datatypeLeavesTotal',
    'countDimensionsGapped': 'datatypeLeavesTotal',
    'datatypeElephino': 'datatypeElephino',
    'datatypeFoldsTotal': 'datatypeFoldsTotal',
    'datatypeLeavesTotal': 'datatypeLeavesTotal',
    'dimensionsTotal': 'datatypeLeavesTotal',
    'dimensionsUnconstrained': 'datatypeLeavesTotal',
    'foldGroups': 'datatypeFoldsTotal',
    'gap1ndex': 'datatypeLeavesTotal',
    'gap1ndexCeiling': 'datatypeElephino',
    'gapRangeStart': 'datatypeElephino',
    'gapsWhere': 'datatypeLeavesTotal',
    'groupsOfFolds': 'datatypeFoldsTotal',
    'indexDimension': 'datatypeLeavesTotal',
    'indexLeaf': 'datatypeLeavesTotal',
    'indexMiniGap': 'datatypeElephino',
    'leaf1ndex': 'datatypeLeavesTotal',
    'leafAbove': 'datatypeLeavesTotal',
    'leafBelow': 'datatypeLeavesTotal',
    'leafConnectee': 'datatypeLeavesTotal',
    'mapShape': 'datatypeLeavesTotal',
    'my': 'datatypeElephino',
    'taskDivisions': 'datatypeLeavesTotal',
    'taskIndex': 'datatypeLeavesTotal',
    'track': 'datatypeElephino',
    }
    RubeGoldBerg = _hackSSOTdatatype[identifier]
    if RubeGoldBerg == 'datatypeElephino':
        return _get_datatype('elephino')
    elif RubeGoldBerg == 'datatypeFoldsTotal':
        return _get_datatype('foldsTotal')
    elif RubeGoldBerg == 'datatypeLeavesTotal':
        return _get_datatype('leavesTotal')
    raise Exception("Dude, you forgot to set a value in `hackSSOTdatatype`.")
