from _typeshed import Incomplete

_already_initialized: bool
logger: Incomplete

def init_all() -> None:
    """Execute all `numba_extensions` entry points with the name `init`

    If extensions have already been initialized, this function does nothing.
    """
