import abc
from _typeshed import Incomplete
from numba import cloudpickle as cloudpickle

def _rebuild_reduction(cls, *args):
    """
    Global hook to rebuild a given class from its __reduce__ arguments.
    """

_unpickled_memo: Incomplete

def _numba_unpickle(address, bytedata, hashed):
    """Used by `numba_unpickle` from _helperlib.c

    Parameters
    ----------
    address : int
    bytedata : bytes
    hashed : bytes

    Returns
    -------
    obj : object
        unpickled object
    """
def dumps(obj):
    """Similar to `pickle.dumps()`. Returns the serialized object in bytes.
    """
def runtime_build_excinfo_struct(static_exc, exc_args): ...

loads: Incomplete

class _CustomPickled:
    """A wrapper for objects that must be pickled with `NumbaPickler`.

    Standard `pickle` will pick up the implementation registered via `copyreg`.
    This will spawn a `NumbaPickler` instance to serialize the data.

    `NumbaPickler` overrides the handling of this type so as not to spawn a
    new pickler for the object when it is already being pickled by a
    `NumbaPickler`.
    """
    __slots__: Incomplete
    ctor: Incomplete
    states: Incomplete
    def __init__(self, ctor, states) -> None: ...
    def _reduce(self): ...
    @classmethod
    def _rebuild(cls, ctor, states): ...

def _unpickle__CustomPickled(serialized):
    """standard unpickling for `_CustomPickled`.

    Uses `NumbaPickler` to load.
    """
def _pickle__CustomPickled(cp):
    """standard pickling for `_CustomPickled`.

    Uses `NumbaPickler` to dump.
    """
def custom_reduce(cls, states):
    """For customizing object serialization in `__reduce__`.

    Object states provided here are used as keyword arguments to the
    `._rebuild()` class method.

    Parameters
    ----------
    states : dict
        Dictionary of object states to be serialized.

    Returns
    -------
    result : tuple
        This tuple conforms to the return type requirement for `__reduce__`.
    """
def custom_rebuild(custom_pickled):
    """Customized object deserialization.

    This function is referenced internally by `custom_reduce()`.
    """
def is_serialiable(obj):
    """Check if *obj* can be serialized.

    Parameters
    ----------
    obj : object

    Returns
    --------
    can_serialize : bool
    """
def _no_pickle(obj) -> None: ...
def disable_pickling(typ):
    """This is called on a type to disable pickling
    """

class NumbaPickler(cloudpickle.CloudPickler):
    disabled_types: Incomplete
    def reducer_override(self, obj): ...

def _custom_reduce__custompickled(cp): ...

class ReduceMixin(abc.ABC, metaclass=abc.ABCMeta):
    """A mixin class for objects that should be reduced by the NumbaPickler
    instead of the standard pickler.
    """
    @abc.abstractmethod
    def _reduce_states(self): ...
    @classmethod
    @abc.abstractmethod
    def _rebuild(cls, **kwargs): ...
    def _reduce_class(self): ...
    def __reduce__(self): ...

class PickleCallableByPath:
    """Wrap a callable object to be pickled by path to workaround limitation
    in pickling due to non-pickleable objects in function non-locals.

    Note:
    - Do not use this as a decorator.
    - Wrapped object must be a global that exist in its parent module and it
      can be imported by `from the_module import the_object`.

    Usage:

    >>> def my_fn(x):
    >>>     ...
    >>> wrapped_fn = PickleCallableByPath(my_fn)
    >>> # refer to `wrapped_fn` instead of `my_fn`
    """
    _fn: Incomplete
    def __init__(self, fn) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def __reduce__(self): ...
    @classmethod
    def _rebuild(cls, modname, fn_path): ...
