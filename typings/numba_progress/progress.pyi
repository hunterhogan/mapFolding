from .numba_atomic import atomic_add as atomic_add, atomic_xchg as atomic_xchg
from _typeshed import Incomplete
from numba import types
from numba.extending import lower_getattr as lower_getattr, lower_setattr as lower_setattr, models

def is_notebook():
    """Determine if we're running within an IPython kernel

    >>> is_notebook()
    False
    """

class ProgressBar:
    """
    Wraps the tqdm progress bar enabling it to be updated from within a numba nopython function.
    It works by spawning a separate thread that updates the tqdm progress bar based on an atomic counter which can be
    accessed within the numba function. The progress bar works with parallel as well as sequential numba functions.
    
    Note: As this Class contains python objects not useable or convertable into numba, it will be boxed as a
    proxy object, that only exposes the minimum subset of functionality to update the progress bar. Attempts
    to return or create a ProgressBar within a numba function will result in an error.

    Parameters
    ----------
    file: `io.TextIOWrapper` or `io.StringIO`, optional
        Specifies where to output the progress messages
        (default: sys.stdout). Uses `file.write(str)` and `file.flush()`
        methods.  For encoding, see `write_bytes`.
    update_interval: float, optional
        The interval in seconds used by the internal thread to check for updates [default: 0.1].
    notebook: bool, optional
        If set, forces or forbits the use of the notebook progress bar. By default the best progress bar will be
        determined automatically.
    dynamic_ncols: bool, optional
        If true, the number of columns (the width of the progress bar) is constantly adjusted. This improves the
        output of the notebook progress bar a lot.
    kwargs: dict-like, optional
        Addtional parameters passed to the tqdm class. See https://github.com/tqdm/tqdm for a documentation of
        the available parameters. Noteable exceptions are the parameters:
            - file is redefined above (see above)
            - iterable is not available because it would not make sense here
            - dynamic_ncols is defined above
    """
    _last_value: int
    _tqdm: Incomplete
    hook: Incomplete
    _updater_thread: Incomplete
    _exit_event: Incomplete
    update_interval: Incomplete
    def __init__(self, file: Incomplete | None = None, update_interval: float = 0.1, notebook: Incomplete | None = None, dynamic_ncols: bool = True, **kwargs) -> None: ...
    _timer: Incomplete
    def _start(self) -> None: ...
    def close(self) -> None: ...
    @property
    def n(self): ...
    def set(self, n: int = 0) -> None: ...
    def update(self, n: int = 1) -> None: ...
    def _update_tqdm(self) -> None: ...
    def _update_function(self) -> None:
        """Background thread for updating the progress bar.
        """
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...

class ProgressBarTypeImpl(types.Type):
    def __init__(self) -> None: ...

ProgressBarType: Incomplete

def typeof_index(val, c): ...

class ProgressBarModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None: ...

def get_value(progress_bar): ...
def unbox_progressbar(typ, obj, c):
    """
    Convert a ProgressBar to it's native representation (proxy object)
    """
def box_progressbar(typ, val, c) -> None: ...
def _ol_update(self, n: int = 1):
    """
    Numpy implementation of the update method.
    """
def _ol_set(self, n: int = 0):
    """
    Numpy implementation of the update method.
    """
