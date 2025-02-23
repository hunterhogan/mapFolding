from _typeshed import Incomplete
from ctypes import Structure, Union
from numba import literal_unroll as literal_unroll
from numba.core import errors as errors, types as types
from numba.core.extending import intrinsic as intrinsic, overload as overload, overload_method as overload_method, register_jitable as register_jitable
from numba.core.unsafe.bytes import grab_byte as grab_byte, grab_uint64_t as grab_uint64_t
from numba.cpython.randomimpl import const_int as const_int, get_next_int as get_next_int, get_next_int32 as get_next_int32, get_state_ptr as get_state_ptr
from typing import NamedTuple

_hash_width: Incomplete
_Py_hash_t: Incomplete
_Py_uhash_t: Incomplete
_PyHASH_INF: Incomplete
_PyHASH_NAN: Incomplete
_PyHASH_MODULUS: Incomplete
_PyHASH_BITS: Incomplete
_PyHASH_MULTIPLIER: int
_PyHASH_IMAG = _PyHASH_MULTIPLIER
_PyLong_SHIFT: Incomplete
_Py_HASH_CUTOFF: Incomplete
_Py_hashfunc_name: Incomplete

def _defer_hash(hash_func) -> None: ...
def ol_defer_hash(obj, hash_func): ...
def hash_overload(obj): ...
def process_return(val): ...
def _Py_HashDouble(v): ...
def _fpext(tyctx, val): ...
def _prng_random_hash(tyctx): ...
def _long_impl(val): ...
def int_hash(val): ...
def float_hash(val): ...
def complex_hash(val): ...

_PyHASH_XXPRIME_1: Incomplete
_PyHASH_XXPRIME_2: Incomplete
_PyHASH_XXPRIME_5: Incomplete

def _PyHASH_XXROTATE(x): ...
def _tuple_hash(tup): ...
def tuple_hash(val): ...

class FNV(Structure):
    _fields_: Incomplete

class SIPHASH(Structure):
    _fields_: Incomplete

class DJBX33A(Structure):
    _fields_: Incomplete

class EXPAT(Structure):
    _fields_: Incomplete

class _Py_HashSecret_t(Union):
    _fields_: Incomplete

class _hashsecret_entry(NamedTuple):
    symbol: Incomplete
    value: Incomplete

def _build_hashsecret():
    '''Read hash secret from the Python process

    Returns
    -------
    info : dict
        - keys are "djbx33a_suffix", "siphash_k0", siphash_k1".
        - values are the namedtuple[symbol:str, value:int]
    '''

_hashsecret: Incomplete
msg: str

def _ROTATE(x, b): ...
def _HALF_ROUND(a, b, c, d, s, t): ...
def _SINGLE_ROUND(v0, v1, v2, v3): ...
def _DOUBLE_ROUND(v0, v1, v2, v3): ...
def _gen_siphash(alg): ...

_siphash13: Incomplete
_siphash24: Incomplete
_siphasher: Incomplete

def _inject_hashsecret_read(tyctx, name):
    """Emit code to load the hashsecret.
    """
def _load_hashsecret(name): ...
def _impl_load_hashsecret(name): ...
def _Py_HashBytes(val, _len): ...
def unicode_hash(val): ...
