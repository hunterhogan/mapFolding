from _typeshed import Incomplete
from numba.np.numpy_support import from_dtype as from_dtype

_WARPSIZE: int
_NUMWARPS: int

def _gpu_reduce_factory(fn, nbtype): ...

class Reduce:
    """Create a reduction object that reduces values using a given binary
    function. The binary function is compiled once and cached inside this
    object. Keeping this object alive will prevent re-compilation.
    """
    _cache: Incomplete
    _functor: Incomplete
    def __init__(self, functor) -> None:
        """
        :param functor: A function implementing a binary operation for
                        reduction. It will be compiled as a CUDA device
                        function using ``cuda.jit(device=True)``.
        """
    def _compile(self, dtype): ...
    def __call__(self, arr, size: Incomplete | None = None, res: Incomplete | None = None, init: int = 0, stream: int = 0):
        """Performs a full reduction.

        :param arr: A host or device array.
        :param size: Optional integer specifying the number of elements in
                    ``arr`` to reduce. If this parameter is not specified, the
                    entire array is reduced.
        :param res: Optional device array into which to write the reduction
                    result to. The result is written into the first element of
                    this array. If this parameter is specified, then no
                    communication of the reduction output takes place from the
                    device to the host.
        :param init: Optional initial value for the reduction, the type of which
                    must match ``arr.dtype``.
        :param stream: Optional CUDA stream in which to perform the reduction.
                    If no stream is specified, the default stream of 0 is
                    used.
        :return: If ``res`` is specified, ``None`` is returned. Otherwise, the
                result of the reduction is returned.
        """
