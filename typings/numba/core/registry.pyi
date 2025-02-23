from _typeshed import Incomplete
from numba.core import cpu as cpu, dispatcher as dispatcher, typing as typing, utils as utils
from numba.core.descriptors import TargetDescriptor as TargetDescriptor

class CPUTarget(TargetDescriptor):
    options = cpu.CPUTargetOptions
    def _toplevel_target_context(self): ...
    def _toplevel_typing_context(self): ...
    @property
    def target_context(self):
        """
        The target context for CPU targets.
        """
    @property
    def typing_context(self):
        """
        The typing context for CPU targets.
        """

cpu_target: Incomplete

class CPUDispatcher(dispatcher.Dispatcher):
    targetdescr = cpu_target

class DelayedRegistry(utils.UniqueDict):
    """
    A unique dictionary but with deferred initialisation of the values.

    Attributes
    ----------
    ondemand:

        A dictionary of key -> value, where value is executed
        the first time it is is used.  It is used for part of a deferred
        initialization strategy.
    """
    ondemand: Incomplete
    key_type: Incomplete
    value_type: Incomplete
    _type_check: Incomplete
    def __init__(self, *args, **kws) -> None: ...
    def __getitem__(self, item): ...
    def __setitem__(self, key, value) -> None: ...
