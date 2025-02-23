from _typeshed import Incomplete
from numba import _dynfunc as _dynfunc

class Environment(_dynfunc.Environment):
    """Stores globals and constant pyobjects for runtime.

    It is often needed to convert b/w nopython objects and pyobjects.
    """
    __slots__: Incomplete
    _memo: Incomplete
    @classmethod
    def from_fndesc(cls, fndesc): ...
    def can_cache(self): ...
    def __reduce__(self): ...
    def __del__(self) -> None: ...
    def __repr__(self) -> str: ...

def _rebuild_env(modname, consts, env_name): ...
def lookup_environment(env_name):
    """Returns the Environment object for the given name;
    or None if not found
    """
