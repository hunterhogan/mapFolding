from _typeshed import Incomplete
from numba.core.registry import DelayedRegistry as DelayedRegistry
from numba.np.ufunc import _internal as _internal, dufunc as dufunc, gufunc as gufunc
from numba.np.ufunc.parallel import ParallelGUFuncBuilder as ParallelGUFuncBuilder, ParallelUFuncBuilder as ParallelUFuncBuilder

class _BaseVectorize:
    @classmethod
    def get_identity(cls, kwargs): ...
    @classmethod
    def get_cache(cls, kwargs): ...
    @classmethod
    def get_writable_args(cls, kwargs): ...
    @classmethod
    def get_target_implementation(cls, kwargs): ...

class Vectorize(_BaseVectorize):
    target_registry: Incomplete
    def __new__(cls, func, **kws): ...

class GUVectorize(_BaseVectorize):
    target_registry: Incomplete
    def __new__(cls, func, signature, **kws): ...

def vectorize(ftylist_or_function=(), **kws):
    '''vectorize(ftylist_or_function=(), target=\'cpu\', identity=None, **kws)

    A decorator that creates a NumPy ufunc object using Numba compiled
    code.  When no arguments or only keyword arguments are given,
    vectorize will return a Numba dynamic ufunc (DUFunc) object, where
    compilation/specialization may occur at call-time.

    Args
    -----
    ftylist_or_function: function or iterable

        When the first argument is a function, signatures are dealt
        with at call-time.

        When the first argument is an iterable of type signatures,
        which are either function type object or a string describing
        the function type, signatures are finalized at decoration
        time.

    Keyword Args
    ------------

    target: str
            A string for code generation target.  Default to "cpu".

    identity: int, str, or None
        The identity (or unit) value for the element-wise function
        being implemented.  Allowed values are None (the default), 0, 1,
        and "reorderable".

    cache: bool
        Turns on caching.


    Returns
    --------

    A NumPy universal function

    Examples
    -------
        @vectorize([\'float32(float32, float32)\',
                    \'float64(float64, float64)\'], identity=0)
        def sum(a, b):
            return a + b

        @vectorize
        def sum(a, b):
            return a + b

        @vectorize(identity=1)
        def mul(a, b):
            return a * b

    '''
def guvectorize(*args, **kwargs):
    '''guvectorize(ftylist, signature, target=\'cpu\', identity=None, **kws)

    A decorator to create NumPy generalized-ufunc object from Numba compiled
    code.

    Args
    -----
    ftylist: iterable
        An iterable of type signatures, which are either
        function type object or a string describing the
        function type.

    signature: str
        A NumPy generalized-ufunc signature.
        e.g. "(m, n), (n, p)->(m, p)"

    identity: int, str, or None
        The identity (or unit) value for the element-wise function
        being implemented.  Allowed values are None (the default), 0, 1,
        and "reorderable".

    cache: bool
        Turns on caching.

    writable_args: tuple
        a tuple of indices of input variables that are writable.

    target: str
            A string for code generation target.  Defaults to "cpu".

    Returns
    --------

    A NumPy generalized universal-function

    Example
    -------
        @guvectorize([\'void(int32[:,:], int32[:,:], int32[:,:])\',
                      \'void(float32[:,:], float32[:,:], float32[:,:])\'],
                      \'(x, y),(x, y)->(x, y)\')
        def add_2d_array(a, b, c):
            for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                    c[i, j] = a[i, j] + b[i, j]

    '''
