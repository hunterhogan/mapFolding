from _typeshed import Incomplete

IS_WIN32: Incomplete
numpy_version: Incomplete
_ufunc_db: Incomplete

def _lazy_init_db() -> None: ...
def get_ufuncs():
    """obtain a list of supported ufuncs in the db"""
def get_ufunc_info(ufunc_key):
    '''get the lowering information for the ufunc with key ufunc_key.

    The lowering information is a dictionary that maps from a numpy
    loop string (as given by the ufunc types attribute) to a function
    that handles code generation for a scalar version of the ufunc
    (that is, generates the "per element" operation").

    raises a KeyError if the ufunc is not in the ufunc_db
    '''
def _fill_ufunc_db(ufunc_db) -> None: ...
