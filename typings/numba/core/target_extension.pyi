import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from numba.core.decorators import jit as jit
from numba.core.errors import InternalTargetMismatchError as InternalTargetMismatchError, NonexistentTargetError as NonexistentTargetError
from numba.core.registry import CPUDispatcher as CPUDispatcher, DelayedRegistry as DelayedRegistry

_active_context: Incomplete
_active_context_default: str

class _TargetRegistry(DelayedRegistry):
    def __getitem__(self, item): ...

target_registry: Incomplete
jit_registry: Incomplete

class target_override:
    """Context manager to temporarily override the current target with that
       prescribed."""
    _orig_target: Incomplete
    target: Incomplete
    def __init__(self, name) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, ty: type[BaseException] | None, val: BaseException | None, tb: types.TracebackType | None) -> None: ...

def current_target():
    """Returns the current target
    """
def get_local_target(context):
    """
    Gets the local target from the call stack if available and the TLS
    override if not.
    """
def resolve_target_str(target_str):
    """Resolves a target specified as a string to its Target class."""
def resolve_dispatcher_from_str(target_str):
    """Returns the dispatcher associated with a target string"""
def _get_local_target_checked(tyctx, hwstr, reason):
    """Returns the local target if it is compatible with the given target
    name during a type resolution; otherwise, raises an exception.

    Parameters
    ----------
    tyctx: typing context
    hwstr: str
        target name to check against
    reason: str
        Reason for the resolution. Expects a noun.
    Returns
    -------
    target_hw : Target

    Raises
    ------
    InternalTargetMismatchError
    """

class JitDecorator(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def __call__(self): ...

class Target(ABC):
    """ Implements a target """
    @classmethod
    def inherits_from(cls, other):
        """Returns True if this target inherits from 'other' False otherwise"""

class Generic(Target):
    """Mark the target as generic, i.e. suitable for compilation on
    any target. All must inherit from this.
    """
class CPU(Generic):
    """Mark the target as CPU.
    """
class GPU(Generic):
    """Mark the target as GPU, i.e. suitable for compilation on a GPU
    target.
    """
class CUDA(GPU):
    """Mark the target as CUDA.
    """
class NPyUfunc(Target):
    """Mark the target as a ufunc
    """

dispatcher_registry: Incomplete
cpu_target: Incomplete
