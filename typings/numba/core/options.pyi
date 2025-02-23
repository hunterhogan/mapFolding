from _typeshed import Incomplete
from numba.core import config as config, utils as utils
from numba.core.targetconfig import Option as Option, TargetConfig as TargetConfig

class TargetOptions:
    """Target options maps user options from decorators to the
    ``numba.core.compiler.Flags`` used by lowering and target context.
    """
    class Mapping:
        flag_name: Incomplete
        apply: Incomplete
        def __init__(self, flag_name, apply=...) -> None: ...
    def finalize(self, flags, options) -> None:
        """Subclasses can override this method to make target specific
        customizations of default flags.

        Parameters
        ----------
        flags : Flags
        options : dict
        """
    @classmethod
    def parse_as_flags(cls, flags, options):
        """Parse target options defined in ``options`` and set ``flags``
        accordingly.

        Parameters
        ----------
        flags : Flags
        options : dict
        """
    def _apply(self, flags, options) -> None: ...
_mapping = TargetOptions.Mapping

class DefaultOptions:
    """Defines how user-level target options are mapped to the target flags.
    """
    nopython: Incomplete
    forceobj: Incomplete
    looplift: Incomplete
    _nrt: Incomplete
    debug: Incomplete
    boundscheck: Incomplete
    nogil: Incomplete
    writable_args: Incomplete
    no_rewrites: Incomplete
    no_cpython_wrapper: Incomplete
    no_cfunc_wrapper: Incomplete
    parallel: Incomplete
    fastmath: Incomplete
    error_model: Incomplete
    inline: Incomplete
    forceinline: Incomplete
    _dbg_extend_lifetimes: Incomplete
    _dbg_optnone: Incomplete

def include_default_options(*args):
    """Returns a mixin class with a subset of the options

    Parameters
    ----------
    *args : str
        Option names to include.
    """
