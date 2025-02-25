from _typeshed import Incomplete
from numba.core import cgutils as cgutils, config as config, errors as errors, types as types
from numba.core.typing import signature as signature
from numba.extending import intrinsic as intrinsic, overload as overload
from numba.np.numpy_support import as_dtype as as_dtype
from numba.np.ufunc import ufuncbuilder as ufuncbuilder
from numba.np.ufunc.wrappers import _wrapper_info as _wrapper_info

_IS_OSX: Incomplete
_IS_LINUX: Incomplete
_IS_WINDOWS: Incomplete

def get_thread_count():
    """
    Gets the available thread count.
    """

NUM_THREADS: Incomplete

def build_gufunc_kernel(library, ctx, info, sig, inner_ndim):
    """Wrap the original CPU ufunc/gufunc with a parallel dispatcher.
    This function will wrap gufuncs and ufuncs something like.

    Args
    ----
    ctx
        numba's codegen context

    info: (library, env, name)
        inner function info

    sig
        type signature of the gufunc

    inner_ndim
        inner dimension of the gufunc (this is len(sig.args) in the case of a
        ufunc)

    Returns
    -------
    wrapper_info : (library, env, name)
        The info for the gufunc wrapper.

    Details
    -------

    The kernel signature looks like this:

    void kernel(char **args, npy_intp *dimensions, npy_intp* steps, void* data)

    args - the input arrays + output arrays
    dimensions - the dimensions of the arrays
    steps - the step size for the array (this is like sizeof(type))
    data - any additional data

    The parallel backend then stages multiple calls to this kernel concurrently
    across a number of threads. Practically, for each item of work, the backend
    duplicates `dimensions` and adjusts the first entry to reflect the size of
    the item of work, it also forms up an array of pointers into the args for
    offsets to read/write from/to with respect to its position in the items of
    work. This allows the same kernel to be used for each item of work, with
    simply adjusted reads/writes/domain sizes and is safe by virtue of the
    domain partitioning.

    NOTE: The execution backend is passed the requested thread count, but it can
    choose to ignore it (TBB)!
    """

class ParallelUFuncBuilder(ufuncbuilder.UFuncBuilder):
    def build(self, cres, sig): ...

def build_ufunc_wrapper(library, ctx, fname, signature, cres): ...

class ParallelGUFuncBuilder(ufuncbuilder.GUFuncBuilder):
    def __init__(self, py_func, signature, identity: Incomplete | None = None, cache: bool = False, targetoptions={}, writable_args=()) -> None: ...
    def build(self, cres):
        """
        Returns (dtype numbers, function ptr, EnvironmentObject)
        """

def build_gufunc_wrapper(py_func, cres, sin, sout, cache, is_parfors):
    """Build gufunc wrapper for the given arguments.
    The *is_parfors* is a boolean indicating whether the gufunc is being
    built for use as a ParFors kernel. This changes codegen and caching
    behavior.
    """

_backend_init_thread_lock: Incomplete
_windows: Incomplete

class _nop:
    """A no-op contextmanager
    """
    def __enter__(self) -> None: ...
    def __exit__(self, *args) -> None: ...

_backend_init_process_lock: Incomplete

def _set_init_process_lock() -> None: ...

_is_initialized: bool
_threading_layer: Incomplete

def threading_layer():
    """
    Get the name of the threading layer in use for parallel CPU targets
    """
def _check_tbb_version_compatible() -> None:
    """
    Checks that if TBB is present it is of a compatible version.
    """
def _launch_threads(): ...
def _load_threading_functions(lib) -> None: ...
def gen_snt_check(): ...

snt_check: Incomplete

def ol_snt_check(n): ...
def set_num_threads(n: int) -> None:
    """
    Set the number of threads to use for parallel execution.

    By default, all :obj:`numba.config.NUMBA_NUM_THREADS` threads are used.

    This functionality works by masking out threads that are not used.
    Therefore, the number of threads *n* must be less than or equal to
    :obj:`~.NUMBA_NUM_THREADS`, the total number of threads that are launched.
    See its documentation for more details.

    This function can be used inside of a jitted function.

    Parameters
    ----------
    n: The number of threads. Must be between 1 and NUMBA_NUM_THREADS.

    See Also
    --------
    get_num_threads, numba.config.NUMBA_NUM_THREADS,
    numba.config.NUMBA_DEFAULT_NUM_THREADS, :envvar:`NUMBA_NUM_THREADS`

    """
def ol_set_num_threads(n): ...
def get_num_threads() -> int:
    """
    Get the number of threads used for parallel execution.

    By default (if :func:`~.set_num_threads` is never called), all
    :obj:`numba.config.NUMBA_NUM_THREADS` threads are used.

    This number is less than or equal to the total number of threads that are
    launched, :obj:`numba.config.NUMBA_NUM_THREADS`.

    This function can be used inside of a jitted function.

    Returns
    -------
    The number of threads.

    See Also
    --------
    set_num_threads, numba.config.NUMBA_NUM_THREADS,
    numba.config.NUMBA_DEFAULT_NUM_THREADS, :envvar:`NUMBA_NUM_THREADS`

    """
def ol_get_num_threads(): ...
def _iget_num_threads(typingctx): ...
def get_thread_id():
    """
    Returns a unique ID for each thread in the range 0 (inclusive)
    to :func:`~.get_num_threads` (exclusive).
    """
def ol_get_thread_id(): ...
def _iget_thread_id(typingctx): ...

_DYLD_WORKAROUND_SET: Incomplete
_DYLD_WORKAROUND_VAL: Incomplete

def set_parallel_chunksize(n): ...
def get_parallel_chunksize(): ...
def ol_set_parallel_chunksize(n): ...
def ol_get_parallel_chunksize(): ...
