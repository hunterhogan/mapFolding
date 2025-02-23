import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from numba.core import config as config, ir as ir
from typing import Callable, Sequence

coverage_available: bool

def get_active_coverage():
    """Get active coverage instance or return None if not found.
    """

_the_registry: Callable[[], NotifyLocBase | None]

def get_registered_loc_notify() -> Sequence['NotifyLocBase']:
    """
    Returns a list of the registered NotifyLocBase instances.
    """
def _get_coverage_data():
    """
    Make a singleton ``CoverageData``.
    Avoid writing to disk. Other processes can corrupt the file.
    """

class NotifyLocBase(ABC, metaclass=abc.ABCMeta):
    """Interface for notifying visiting of a ``numba.core.ir.Loc``.
    """
    @abstractmethod
    def notify(self, loc: ir.Loc) -> None: ...
    @abstractmethod
    def close(self) -> None: ...

class NotifyCompilerCoverage(NotifyLocBase):
    '''
    Use to notify ``coverage`` about compiled lines.

    The compiled lines are under the "numba_compiled" context in the coverage
    data.
    '''
    _arcs_data: Incomplete
    def __init__(self) -> None: ...
    def notify(self, loc: ir.Loc): ...
    def close(self) -> None: ...

def _register_coverage_notifier(): ...
