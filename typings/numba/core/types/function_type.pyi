import abc
from .abstract import Type
from _typeshed import Incomplete
from abc import ABC, abstractmethod

__all__ = ['FunctionType', 'UndefinedFunctionType', 'FunctionPrototype', 'WrapperAddressProtocol', 'CompileResultWAP']

class FunctionType(Type):
    """
    First-class function type.
    """
    cconv: Incomplete
    nargs: Incomplete
    signature: Incomplete
    ftype: Incomplete
    _key: Incomplete
    def __init__(self, signature) -> None: ...
    @property
    def key(self): ...
    @property
    def name(self): ...
    def is_precise(self): ...
    def get_precise(self): ...
    def dump(self, tab: str = '') -> None: ...
    def get_call_type(self, context, args, kws): ...
    def check_signature(self, other_sig):
        """Return True if signatures match (up to being precise).
        """
    def unify(self, context, other): ...

class UndefinedFunctionType(FunctionType):
    _counter: int
    dispatchers: Incomplete
    def __init__(self, nargs, dispatchers) -> None: ...
    def get_precise(self):
        """
        Return precise function type if possible.
        """

class FunctionPrototype(Type):
    """
    Represents the prototype of a first-class function type.
    Used internally.
    """
    cconv: Incomplete
    rtype: Incomplete
    atypes: Incomplete
    def __init__(self, rtype, atypes) -> None: ...
    @property
    def key(self): ...

class WrapperAddressProtocol(ABC, metaclass=abc.ABCMeta):
    """Base class for Wrapper Address Protocol.

    Objects that inherit from the WrapperAddressProtocol can be passed
    as arguments to Numba jit compiled functions where it can be used
    as first-class functions. As a minimum, the derived types must
    implement two methods ``__wrapper_address__`` and ``signature``.
    """
    @abstractmethod
    def __wrapper_address__(self):
        """Return the address of a first-class function.

        Returns
        -------
        addr : int
        """
    @abstractmethod
    def signature(self):
        """Return the signature of a first-class function.

        Returns
        -------
        sig : Signature
          The returned Signature instance represents the type of a
          first-class function that the given WrapperAddressProtocol
          instance represents.
        """

class CompileResultWAP(WrapperAddressProtocol):
    """Wrapper of dispatcher instance compilation result to turn it a
    first-class function.
    """
    cres: Incomplete
    address: Incomplete
    def __init__(self, cres) -> None:
        """
        Parameters
        ----------
        cres : CompileResult
          Specify compilation result of a Numba jit-decorated function
          (that is a value of dispatcher instance ``overloads``
          attribute)
        """
    def dump(self, tab: str = '') -> None: ...
    def __wrapper_address__(self): ...
    def signature(self): ...
    def __call__(self, *args, **kwargs): ...
