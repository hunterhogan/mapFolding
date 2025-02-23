from numba import cuda as cuda, types as types
from numba.core import cgutils as cgutils
from numba.core.errors import NumbaValueError as NumbaValueError, RequireLiteralValue as RequireLiteralValue
from numba.core.extending import overload_attribute as overload_attribute
from numba.core.typing import signature as signature
from numba.cuda import nvvmutils as nvvmutils
from numba.cuda.extending import intrinsic as intrinsic

def _type_grid_function(ndim): ...
def grid(typingctx, ndim):
    """grid(ndim)

    Return the absolute position of the current thread in the entire grid of
    blocks.  *ndim* should correspond to the number of dimensions declared when
    instantiating the kernel. If *ndim* is 1, a single integer is returned.
    If *ndim* is 2 or 3, a tuple of the given number of integers is returned.

    Computation of the first integer is as follows::

        cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    and is similar for the other two indices, but using the ``y`` and ``z``
    attributes.
    """
def gridsize(typingctx, ndim):
    """gridsize(ndim)

    Return the absolute size (or shape) in threads of the entire grid of
    blocks. *ndim* should correspond to the number of dimensions declared when
    instantiating the kernel. If *ndim* is 1, a single integer is returned.
    If *ndim* is 2 or 3, a tuple of the given number of integers is returned.

    Computation of the first integer is as follows::

        cuda.blockDim.x * cuda.gridDim.x

    and is similar for the other two indices, but using the ``y`` and ``z``
    attributes.
    """
def _warpsize(typingctx): ...
def cuda_warpsize(mod):
    """
    The size of a warp. All architectures implemented to date have a warp size
    of 32.
    """
def syncthreads(typingctx):
    """
    Synchronize all threads in the same thread block.  This function implements
    the same pattern as barriers in traditional multi-threaded programming: this
    function waits until all threads in the block call it, at which point it
    returns control to all its callers.
    """
def _syncthreads_predicate(typingctx, predicate, fname): ...
def syncthreads_count(typingctx, predicate):
    """
    syncthreads_count(predicate)

    An extension to numba.cuda.syncthreads where the return value is a count
    of the threads where predicate is true.
    """
def syncthreads_and(typingctx, predicate):
    """
    syncthreads_and(predicate)

    An extension to numba.cuda.syncthreads where 1 is returned if predicate is
    true for all threads or 0 otherwise.
    """
def syncthreads_or(typingctx, predicate):
    """
    syncthreads_or(predicate)

    An extension to numba.cuda.syncthreads where 1 is returned if predicate is
    true for any thread or 0 otherwise.
    """
