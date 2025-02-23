import abc
from _typeshed import Incomplete

__all__ = ['In', 'Out', 'InOut', 'ArgHint', 'wrap_arg']

class ArgHint(metaclass=abc.ABCMeta):
    value: Incomplete
    def __init__(self, value) -> None: ...
    @abc.abstractmethod
    def to_device(self, retr, stream: int = 0):
        """
        :param stream: a stream to use when copying data
        :param retr:
            a list of clean-up work to do after the kernel's been run.
            Append 0-arg lambdas to it!
        :return: a value (usually an `DeviceNDArray`) to be passed to
            the kernel
        """
    @property
    def _numba_type_(self): ...

class In(ArgHint):
    def to_device(self, retr, stream: int = 0): ...

class Out(ArgHint):
    def to_device(self, retr, stream: int = 0): ...

class InOut(ArgHint):
    def to_device(self, retr, stream: int = 0): ...

def wrap_arg(value, default=...): ...
