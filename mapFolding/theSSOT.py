from numpy import integer
from numpy.typing import NDArray
from typing import Any, Callable, Optional, Tuple, Type, TypedDict
from types import ModuleType
import enum
import numba
import numpy
import numpy.typing
import pathlib
import sys

def getAlgorithmSource() -> ModuleType:
    from mapFolding import theDao
    return theDao

def getAlgorithmCallable() -> Callable[..., None]:
    algorithmSource = getAlgorithmSource()
    return algorithmSource.doTheNeedful

def getDispatcherCallable() -> Callable[..., None]:
    from mapFolding import dispatcher
    return dispatcher._countFolds

# NOTE I want this _concept_ to be well implemented and usable everywhere: Python, Numba, Jax, CUDA, idc
class computationState(TypedDict):
    connectionGraph: NDArray[integer[Any]]
    foldGroups: NDArray[integer[Any]]
    gapsWhere: NDArray[integer[Any]]
    mapShape: Tuple[int, ...]
    my: NDArray[integer[Any]]
    track: NDArray[integer[Any]]

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
    dimensionsTotal = enum.auto() # connectionGraph.shape[0] or len(mapShape)
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

from dataclasses import dataclass
from typing import Final

@dataclass()
class DatatypeDefaults():
    """Configuration for numeric datatypes used in computation."""

    def __init__(self, module: str, large: str, medium: str, small: str) -> None:
        self.module = module
        self.large = large
        self.medium = medium
        self.small = small

    def make_dtype(self, datatype: str) -> Any:
        """Convert datatype string to actual type."""
        return eval(f"{self.module}.{datatype}")

    @property
    def dtypeLarge(self) -> Any:
        return self.make_dtype(self.large)

    @property
    def dtypeMedium(self) -> Any:
        return self.make_dtype(self.medium)

    @property
    def dtypeSmall(self) -> Any:
        return self.make_dtype(self.small)

# Global configuration instance
@dataclass(frozen=True)
class Z0Z_strings():
    datatypeLarge: Final[str] = 'int64'
    datatypeMedium: Final[str] = 'uint8'
    datatypeSmall: Final[str] = datatypeMedium
thisSeemsVeryComplicated: Final = Z0Z_strings()

# If I use the dataclass instance, these are transitory variables
datatypeLarge = thisSeemsVeryComplicated.datatypeLarge
datatypeMedium = thisSeemsVeryComplicated.datatypeMedium
datatypeSmall = thisSeemsVeryComplicated.datatypeSmall

dtypeNumpyDefaults: Final = DatatypeDefaults('numpy', datatypeLarge, datatypeMedium, datatypeSmall)

# Use configparser to manage changes to datatypes.
# Actually, wait. I switched to numba typing in the signature because I wanted
# fewer imports because I hoped it would improve speed.
# Perhaps I should switch back to numpy types with implicit numba types
# The integer size should be based on the specific job
datatypeModuleDEFAULT = 'numpy'
# datatypeMedium = datatypeLarge
# datatypeSmall = datatypeMedium

def make_dtype(datatype: str, datatypeModule: Optional[str] = None) -> Type[Any]:
    if datatypeModule is None:
        datatypeModule = datatypeModuleDEFAULT
    return eval(f"{datatypeModule}.{datatype}")

dtypeLarge = make_dtype(datatypeLarge)
dtypeMedium = make_dtype(datatypeMedium)
dtypeSmall = make_dtype(datatypeSmall)

try:
    _pathModule = pathlib.Path(__file__).parent
except NameError:
    _pathModule = pathlib.Path.cwd()

pathJobDEFAULT = _pathModule / "jobs"

if 'google.colab' in sys.modules:
    pathJobDEFAULT = pathlib.Path("/content/drive/MyDrive") / "jobs"
