from collections import defaultdict
from numpy import integer
from types import ModuleType
from typing import Any, Callable, Dict, Final, Optional, Tuple, Type, TypedDict
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

def getAlgorithmSource() -> ModuleType:
    from mapFolding import theDao
    return theDao

def getAlgorithmCallable() -> Callable[..., None]:
    algorithmSource = getAlgorithmSource()
    return algorithmSource.doTheNeedful

def getDispatcherCallable() -> Callable[..., None]:
    from mapFolding import dispatcherNumba
    return dispatcherNumba._countFolds

# NOTE I want this _concept_ to be well implemented and usable everywhere: Python, Numba, Jax, CUDA, idc
class computationState(TypedDict):
    connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]]
    foldGroups: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
    gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
    mapShape: Tuple[int, ...]
    my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
    track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]

@enum.verify(enum.CONTINUOUS, enum.UNIQUE) if sys.version_info >= (3, 11) else lambda x: x
class EnumIndices(enum.IntEnum):
    """Base class for index enums."""
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
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

_datatype = defaultdict(str)
def reportDatatypeLimit(identifier: str, datatype: str, sourGrapes: Optional[bool] = False) -> str:
    global _datatype
    if not _datatype[identifier]:
        _datatype[identifier] = datatype
    elif _datatype[identifier] == datatype:
        pass
    elif sourGrapes:
        raise Exception(f"Datatype is '{_datatype[identifier]}' not '{datatype}', so you can take your ball and go home.")
    return _datatype[identifier]

def setDatatypeModule(datatypeModule: str, sourGrapes: Optional[bool] = False):
    global _datatypeModule
    if not _datatypeModule:
        _datatypeModule = datatypeModule
    elif _datatypeModule == datatypeModule:
        pass
    elif sourGrapes:
        raise Exception(f"Datatype module is '{_datatypeModule}' not '{datatypeModule}', so you can take your ball and go home.")
    return _datatypeModule

def setDatatypeElephino(datatype: str, sourGrapes: Optional[bool] = False):
    return reportDatatypeLimit('elephino', datatype, sourGrapes)

def setDatatypeFoldsTotal(datatype: str, sourGrapes: Optional[bool] = False):
    return reportDatatypeLimit('foldsTotal', datatype, sourGrapes)

def setDatatypeLeavesTotal(datatype: str, sourGrapes: Optional[bool] = False):
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

def _getDatatypeModule():
    global _datatypeModule
    if not _datatypeModule:
        _datatypeModule = _datatypeModuleDEFAULT
    return _datatypeModule

def setInStone(identifier: str):
    return eval(f"{_getDatatypeModule()}.{_get_datatype(identifier)}")

def hackSSOTdtype(identifier: str) -> Type[Any]:
    _hackSSOTdtype={
    'connectionGraph': 'dtypeLeavesTotal',
    'dtypeElephino': 'dtypeElephino',
    'dtypeFoldsTotal': 'dtypeFoldsTotal',
    'dtypeLeavesTotal': 'dtypeLeavesTotal',
    'foldGroups': 'dtypeFoldsTotal',
    'gapsWhere': 'dtypeLeavesTotal',
    'my': 'dtypeElephino',
    'track': 'dtypeElephino',
    }
    Rube = _hackSSOTdtype[identifier]
    if Rube == 'dtypeElephino':
        GoldBerg = setInStone('elephino')
    elif Rube == 'dtypeFoldsTotal':
        GoldBerg = setInStone('foldsTotal')
    elif Rube == 'dtypeLeavesTotal':
        GoldBerg = setInStone('leavesTotal')
    return GoldBerg

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
    'my': 'datatypeElephino',
    'taskDivisions': 'datatypeLeavesTotal',
    'taskIndex': 'datatypeLeavesTotal',
    'track': 'datatypeElephino',
    }
    Rube = _hackSSOTdatatype[identifier]
    if Rube == 'datatypeElephino':
        GoldBerg = _get_datatype('elephino')
    elif Rube == 'datatypeFoldsTotal':
        GoldBerg = _get_datatype('foldsTotal')
    elif Rube == 'datatypeLeavesTotal':
        GoldBerg = _get_datatype('leavesTotal')
    return GoldBerg

try:
    _pathModule = pathlib.Path(__file__).parent
except NameError:
    _pathModule = pathlib.Path.cwd()

pathJobDEFAULT = _pathModule / "jobs"

if 'google.colab' in sys.modules:
    pathJobDEFAULT = pathlib.Path("/content/drive/MyDrive") / "jobs"

# this needs improvement
relativePathSyntheticModules = "syntheticModules"
