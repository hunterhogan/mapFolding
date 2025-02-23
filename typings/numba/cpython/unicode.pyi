from _typeshed import Incomplete
from numba._helperlib import c_helpers as c_helpers
from numba.core import cgutils as cgutils, config as config, types as types
from numba.core.cgutils import is_nonelike as is_nonelike
from numba.core.datamodel import StructModel as StructModel, register_default as register_default
from numba.core.errors import TypingError as TypingError
from numba.core.extending import NativeValue as NativeValue, box as box, intrinsic as intrinsic, make_attribute_wrapper as make_attribute_wrapper, models as models, overload as overload, overload_method as overload_method, register_jitable as register_jitable, register_model as register_model, unbox as unbox
from numba.core.imputils import RefType as RefType, impl_ret_new_ref as impl_ret_new_ref, iternext_impl as iternext_impl, lower_builtin as lower_builtin, lower_cast as lower_cast, lower_constant as lower_constant
from numba.core.pythonapi import PY_UNICODE_1BYTE_KIND as PY_UNICODE_1BYTE_KIND, PY_UNICODE_2BYTE_KIND as PY_UNICODE_2BYTE_KIND, PY_UNICODE_4BYTE_KIND as PY_UNICODE_4BYTE_KIND, PY_UNICODE_WCHAR_KIND as PY_UNICODE_WCHAR_KIND
from numba.core.unsafe.bytes import memcpy_region as memcpy_region
from numba.core.utils import PYVERSION as PYVERSION
from numba.cpython import slicing as slicing
from numba.cpython.hashing import _Py_hash_t as _Py_hash_t
from numba.cpython.unicode_support import _PyUnicode_IsAlpha as _PyUnicode_IsAlpha, _PyUnicode_IsCaseIgnorable as _PyUnicode_IsCaseIgnorable, _PyUnicode_IsCased as _PyUnicode_IsCased, _PyUnicode_IsDecimalDigit as _PyUnicode_IsDecimalDigit, _PyUnicode_IsDigit as _PyUnicode_IsDigit, _PyUnicode_IsLineBreak as _PyUnicode_IsLineBreak, _PyUnicode_IsLowercase as _PyUnicode_IsLowercase, _PyUnicode_IsNumeric as _PyUnicode_IsNumeric, _PyUnicode_IsPrintable as _PyUnicode_IsPrintable, _PyUnicode_IsSpace as _PyUnicode_IsSpace, _PyUnicode_IsTitlecase as _PyUnicode_IsTitlecase, _PyUnicode_IsUppercase as _PyUnicode_IsUppercase, _PyUnicode_IsXidContinue as _PyUnicode_IsXidContinue, _PyUnicode_IsXidStart as _PyUnicode_IsXidStart, _PyUnicode_ToFoldedFull as _PyUnicode_ToFoldedFull, _PyUnicode_ToLowerFull as _PyUnicode_ToLowerFull, _PyUnicode_ToTitleFull as _PyUnicode_ToTitleFull, _PyUnicode_ToUpperFull as _PyUnicode_ToUpperFull, _Py_CARRIAGE_RETURN as _Py_CARRIAGE_RETURN, _Py_ISALNUM as _Py_ISALNUM, _Py_ISALPHA as _Py_ISALPHA, _Py_ISCARRIAGERETURN as _Py_ISCARRIAGERETURN, _Py_ISLINEBREAK as _Py_ISLINEBREAK, _Py_ISLINEFEED as _Py_ISLINEFEED, _Py_ISLOWER as _Py_ISLOWER, _Py_ISSPACE as _Py_ISSPACE, _Py_ISUPPER as _Py_ISUPPER, _Py_LINEFEED as _Py_LINEFEED, _Py_SPACE as _Py_SPACE, _Py_TAB as _Py_TAB, _Py_TOLOWER as _Py_TOLOWER, _Py_TOUPPER as _Py_TOUPPER, _Py_UCS4 as _Py_UCS4

_MAX_UNICODE: int
_BLOOM_WIDTH: Incomplete

class UnicodeModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None: ...

class UnicodeIteratorModel(StructModel):
    def __init__(self, dmm, fe_type) -> None: ...

def compile_time_get_string_data(obj):
    """Get string data from a python string for use at compile-time to embed
    the string data into the LLVM module.
    """
def make_string_from_constant(context, builder, typ, literal_string):
    """
    Get string data by `compile_time_get_string_data()` and return a
    unicode_type LLVM value
    """
def cast_from_literal(context, builder, fromty, toty, val): ...
def constant_unicode(context, builder, typ, pyval): ...
def unbox_unicode_str(typ, obj, c):
    """
    Convert a unicode str object to a native unicode structure.
    """
def box_unicode_str(typ, val, c):
    """
    Convert a native unicode structure to a unicode string
    """
def make_deref_codegen(bitsize): ...
def deref_uint8(typingctx, data, offset): ...
def deref_uint16(typingctx, data, offset): ...
def deref_uint32(typingctx, data, offset): ...
def _malloc_string(typingctx, kind, char_bytes, length, is_ascii):
    """make empty string with data buffer of size alloc_bytes.

    Must set length and kind values for string after it is returned
    """
def _empty_string(kind, length, is_ascii: int = 0): ...
def _get_code_point(a, i): ...
def make_set_codegen(bitsize): ...
def set_uint8(typingctx, data, idx, ch): ...
def set_uint16(typingctx, data, idx, ch): ...
def set_uint32(typingctx, data, idx, ch): ...
def _set_code_point(a, i, ch) -> None: ...
def _pick_kind(kind1, kind2): ...
def _pick_ascii(is_ascii1, is_ascii2): ...
def _kind_to_byte_width(kind): ...
def _cmp_region(a, a_offset, b, b_offset, n): ...
def _codepoint_to_kind(cp):
    """
    Compute the minimum unicode kind needed to hold a given codepoint
    """
def _codepoint_is_ascii(ch):
    """
    Returns true if a codepoint is in the ASCII range
    """
def unicode_len(s): ...
def unicode_eq(a, b): ...
def unicode_ne(a, b): ...
def unicode_lt(a, b): ...
def unicode_gt(a, b): ...
def unicode_le(a, b): ...
def unicode_ge(a, b): ...
def unicode_contains(a, b): ...
def unicode_idx_check_type(ty, name) -> None:
    """Check object belongs to one of specific types
    ty: type
        Type of the object
    name: str
        Name of the object
    """
def unicode_sub_check_type(ty, name) -> None:
    """Check object belongs to unicode type"""
def _bloom_add(mask, ch): ...
def _bloom_check(mask, ch): ...
def _default_find(data, substr, start, end):
    """Left finder."""
def _default_rfind(data, substr, start, end):
    """Right finder."""
def generate_finder(find_func):
    """Generate finder either left or right."""

_find: Incomplete
_rfind: Incomplete

def unicode_find(data, substr, start: Incomplete | None = None, end: Incomplete | None = None):
    """Implements str.find()"""
def unicode_rfind(data, substr, start: Incomplete | None = None, end: Incomplete | None = None):
    """Implements str.rfind()"""
def unicode_rindex(s, sub, start: Incomplete | None = None, end: Incomplete | None = None):
    """Implements str.rindex()"""
def unicode_index(s, sub, start: Incomplete | None = None, end: Incomplete | None = None):
    """Implements str.index()"""
def unicode_partition(data, sep):
    """Implements str.partition()"""
def unicode_count(src, sub, start: Incomplete | None = None, end: Incomplete | None = None): ...
def unicode_rpartition(data, sep):
    """Implements str.rpartition()"""
def _adjust_indices(length, start, end): ...
def unicode_startswith(s, prefix, start: Incomplete | None = None, end: Incomplete | None = None): ...
def unicode_endswith(s, substr, start: Incomplete | None = None, end: Incomplete | None = None): ...
def unicode_expandtabs(data, tabsize: int = 8):
    """Implements str.expandtabs()"""
def unicode_split(a, sep: Incomplete | None = None, maxsplit: int = -1): ...
def generate_rsplit_whitespace_impl(isspace_func):
    """Generate whitespace rsplit func based on either ascii or unicode"""

unicode_rsplit_whitespace_impl: Incomplete
ascii_rsplit_whitespace_impl: Incomplete

def unicode_rsplit(data, sep: Incomplete | None = None, maxsplit: int = -1):
    """Implements str.unicode_rsplit()"""
def unicode_center(string, width, fillchar: str = ' '): ...
def gen_unicode_Xjust(STRING_FIRST): ...
def generate_splitlines_func(is_line_break_func):
    """Generate splitlines performer based on ascii or unicode line breaks."""

_ascii_splitlines: Incomplete
_unicode_splitlines: Incomplete

def unicode_splitlines(data, keepends: bool = False):
    """Implements str.splitlines()"""
def join_list(sep, parts): ...
def unicode_join(sep, parts): ...
def unicode_zfill(string, width): ...
def unicode_strip_left_bound(string, chars): ...
def unicode_strip_right_bound(string, chars): ...
def unicode_strip_types_check(chars) -> None: ...
def _count_args_types_check(arg) -> None: ...
def unicode_lstrip(string, chars: Incomplete | None = None): ...
def unicode_rstrip(string, chars: Incomplete | None = None): ...
def unicode_strip(string, chars: Incomplete | None = None): ...
def normalize_str_idx(idx, length, is_start: bool = True):
    """
    Parameters
    ----------
    idx : int or None
        the index
    length : int
        the string length
    is_start : bool; optional with defaults to True
        Is it the *start* or the *stop* of the slice?

    Returns
    -------
    norm_idx : int
        normalized index
    """
def _normalize_slice_idx_count(arg, slice_len, default):
    """
    Used for unicode_count

    If arg < -slice_len, returns 0 (prevents circle)

    If arg is within slice, e.g -slice_len <= arg < slice_len
    returns its real index via arg % slice_len

    If arg > slice_len, returns arg (in this case count must
    return 0 if it is start index)
    """
def _normalize_slice(typingctx, sliceobj, length):
    """Fix slice object.
    """
def _slice_span(typingctx, sliceobj):
    """Compute the span from the given slice object.
    """
def _strncpy(dst, dst_offset, src, src_offset, n) -> None: ...
def _get_str_slice_view(typingctx, src_t, start_t, length_t):
    """Create a slice of a unicode string using a view of its data to avoid
    extra allocation.
    """
def unicode_getitem(s, idx): ...
def unicode_concat(a, b): ...
def _repeat_impl(str_arg, mult_arg): ...
def unicode_repeat(a, b): ...
def unicode_not(a): ...
def unicode_replace(s, old_str, new_str, count: int = -1): ...
def gen_isAlX(ascii_func, unicode_func): ...

_unicode_is_alnum: Incomplete

def _is_upper(is_lower, is_upper, is_title): ...

_always_false: Incomplete
_ascii_is_upper: Incomplete
_unicode_is_upper: Incomplete

def unicode_isupper(a):
    """
    Implements .isupper()
    """
def unicode_isascii(data):
    """Implements UnicodeType.isascii()"""
def unicode_istitle(data):
    """
    Implements UnicodeType.istitle()
    The algorithm is an approximate translation from CPython:
    https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodeobject.c#L11829-L11885 # noqa: E501
    """
def unicode_islower(data):
    """
    impl is an approximate translation of:
    https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Objects/unicodeobject.c#L11900-L11933    # noqa: E501
    mixed with:
    https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Objects/bytes_methods.c#L131-L156    # noqa: E501
    """
def unicode_isidentifier(data):
    """Implements UnicodeType.isidentifier()"""
def gen_isX(_PyUnicode_IS_func, empty_is_false: bool = True): ...
def case_operation(ascii_func, unicode_func):
    """Generate common case operation performer."""
def _handle_capital_sigma(data, length, idx):
    """This is a translation of the function that handles the capital sigma."""
def _lower_ucs4(code_point, data, length, idx, mapped):
    """This is a translation of the function that lowers a character."""
def _gen_unicode_upper_or_lower(lower): ...

_unicode_upper: Incomplete
_unicode_lower: Incomplete

def _gen_ascii_upper_or_lower(func): ...

_ascii_upper: Incomplete
_ascii_lower: Incomplete

def unicode_lower(data):
    """Implements .lower()"""
def unicode_upper(data):
    """Implements .upper()"""
def _unicode_casefold(data, length, res, maxchars): ...
def _ascii_casefold(data, res) -> None: ...
def unicode_casefold(data):
    """Implements str.casefold()"""
def _unicode_capitalize(data, length, res, maxchars): ...
def _ascii_capitalize(data, res) -> None: ...
def unicode_capitalize(data): ...
def _unicode_title(data, length, res, maxchars):
    """This is a translation of the function that titles a unicode string."""
def _ascii_title(data, res) -> None:
    """ Does .title() on an ASCII string """
def unicode_title(data):
    """Implements str.title()"""
def _ascii_swapcase(data, res) -> None: ...
def _unicode_swapcase(data, length, res, maxchars): ...
def unicode_swapcase(data): ...
def ol_ord(c): ...
def _unicode_char(ch): ...

_out_of_range_msg: Incomplete

def _PyUnicode_FromOrdinal(ordinal): ...
def ol_chr(i): ...
def unicode_str(s): ...
def unicode_repr(s): ...
def integer_str(n): ...
def integer_repr(n): ...
def boolean_str(b): ...
def getiter_unicode(context, builder, sig, args): ...
def iternext_unicode(context, builder, sig, args, result) -> None: ...
