from _typeshed import Incomplete

_HAVE_YAML: bool
IS_WIN32: Incomplete
IS_OSX: Incomplete
MACHINE_BITS: Incomplete
IS_32BITS: Incomplete
PYVERSION: Incomplete
_config_fname: str

def _parse_cc(text):
    """
    Parse CUDA compute capability version string.
    """
def _os_supports_avx():
    """
    Whether the current OS supports AVX, regardless of the CPU.

    This is necessary because the user may be running a very old Linux
    kernel (e.g. CentOS 5) on a recent CPU.
    """

class _OptLevel(int):
    '''This class holds the "optimisation level" set in `NUMBA_OPT`. As this env
    var can be an int or a string, but is almost always interpreted as an int,
    this class subclasses int so as to get the common behaviour but stores the
    actual value as a `_raw_value` member. The value "max" is a special case
    and the property `is_opt_max` can be queried to find if the optimisation
    level (supplied value at construction time) is "max".'''
    def __new__(cls, *args, **kwargs): ...
    @property
    def is_opt_max(self):
        '''Returns True if the the optimisation level is "max" False
        otherwise.'''
    def __repr__(self) -> str: ...

def _process_opt_level(opt_level): ...

class _EnvReloader:
    def __init__(self) -> None: ...
    old_environ: Incomplete
    def reset(self) -> None: ...
    def update(self, force: bool = False) -> None: ...
    def validate(self) -> None: ...
    def process_environ(self, environ): ...

_env_reloader: Incomplete

def reload_config() -> None:
    """
    Reload the configuration from environment variables, if necessary.
    """
