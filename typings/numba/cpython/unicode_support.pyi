from _typeshed import Incomplete
from enum import IntEnum
from numba.core import cgutils as cgutils, config as config, types as types
from numba.core.errors import TypingError as TypingError
from numba.core.extending import intrinsic as intrinsic, overload as overload, register_jitable as register_jitable
from numba.core.imputils import impl_ret_untracked as impl_ret_untracked
from typing import NamedTuple

class typerecord(NamedTuple):
    upper: Incomplete
    lower: Incomplete
    title: Incomplete
    decimal: Incomplete
    digit: Incomplete
    flags: Incomplete

_Py_UCS4: Incomplete
_Py_TAB: int
_Py_LINEFEED: int
_Py_CARRIAGE_RETURN: int
_Py_SPACE: int

class _PyUnicode_TyperecordMasks(IntEnum):
    ALPHA_MASK = 1
    DECIMAL_MASK = 2
    DIGIT_MASK = 4
    LOWER_MASK = 8
    LINEBREAK_MASK = 16
    SPACE_MASK = 32
    TITLE_MASK = 64
    UPPER_MASK = 128
    XID_START_MASK = 256
    XID_CONTINUE_MASK = 512
    PRINTABLE_MASK = 1024
    NUMERIC_MASK = 2048
    CASE_IGNORABLE_MASK = 4096
    CASED_MASK = 8192
    EXTENDED_CASE_MASK = 16384

def _PyUnicode_gettyperecord(a) -> None: ...
def _gettyperecord_impl(typingctx, codepoint):
    """
    Provides the binding to numba_gettyperecord, returns a `typerecord`
    namedtuple of properties from the codepoint.
    """
def gettyperecord_impl(a):
    """
    Provides a _PyUnicode_gettyperecord binding, for convenience it will accept
    single character strings and code points.
    """
def _PyUnicode_ExtendedCase(typingctx, index):
    """
    Accessor function for the _PyUnicode_ExtendedCase array, binds to
    numba_get_PyUnicode_ExtendedCase which wraps the array and does the lookup
    """
def _PyUnicode_ToTitlecase(ch): ...
def _PyUnicode_IsTitlecase(ch): ...
def _PyUnicode_IsXidStart(ch): ...
def _PyUnicode_IsXidContinue(ch): ...
def _PyUnicode_ToDecimalDigit(ch): ...
def _PyUnicode_ToDigit(ch): ...
def _PyUnicode_IsNumeric(ch): ...
def _PyUnicode_IsPrintable(ch): ...
def _PyUnicode_IsLowercase(ch): ...
def _PyUnicode_IsUppercase(ch): ...
def _PyUnicode_IsLineBreak(ch): ...
def _PyUnicode_ToUppercase(ch) -> None: ...
def _PyUnicode_ToLowercase(ch) -> None: ...
def _PyUnicode_ToLowerFull(ch, res): ...
def _PyUnicode_ToTitleFull(ch, res): ...
def _PyUnicode_ToUpperFull(ch, res): ...
def _PyUnicode_ToFoldedFull(ch, res): ...
def _PyUnicode_IsCased(ch): ...
def _PyUnicode_IsCaseIgnorable(ch): ...
def _PyUnicode_IsDigit(ch): ...
def _PyUnicode_IsDecimalDigit(ch): ...
def _PyUnicode_IsSpace(ch): ...
def _PyUnicode_IsAlpha(ch): ...

class _PY_CTF(IntEnum):
    LOWER = 1
    UPPER = 2
    ALPHA = ...
    DIGIT = 4
    ALNUM = ...
    SPACE = 8
    XDIGIT = 16

_Py_ctype_table: Incomplete
_Py_ctype_tolower: Incomplete
_Py_ctype_toupper: Incomplete

class _PY_CTF_LB(IntEnum):
    LINE_BREAK = 1
    LINE_FEED = 2
    CARRIAGE_RETURN = 4

_Py_ctype_islinebreak: Incomplete

def _Py_CHARMASK(ch):
    """
    Equivalent to the CPython macro `Py_CHARMASK()`, masks off all but the
    lowest 256 bits of ch.
    """
def _Py_TOUPPER(ch):
    """
    Equivalent to the CPython macro `Py_TOUPPER()` converts an ASCII range
    code point to the upper equivalent
    """
def _Py_TOLOWER(ch):
    """
    Equivalent to the CPython macro `Py_TOLOWER()` converts an ASCII range
    code point to the lower equivalent
    """
def _Py_ISLOWER(ch):
    """
    Equivalent to the CPython macro `Py_ISLOWER()`
    """
def _Py_ISUPPER(ch):
    """
    Equivalent to the CPython macro `Py_ISUPPER()`
    """
def _Py_ISALPHA(ch):
    """
    Equivalent to the CPython macro `Py_ISALPHA()`
    """
def _Py_ISDIGIT(ch):
    """
    Equivalent to the CPython macro `Py_ISDIGIT()`
    """
def _Py_ISXDIGIT(ch):
    """
    Equivalent to the CPython macro `Py_ISXDIGIT()`
    """
def _Py_ISALNUM(ch):
    """
    Equivalent to the CPython macro `Py_ISALNUM()`
    """
def _Py_ISSPACE(ch):
    """
    Equivalent to the CPython macro `Py_ISSPACE()`
    """
def _Py_ISLINEBREAK(ch):
    """Check if character is ASCII line break"""
def _Py_ISLINEFEED(ch):
    """Check if character is line feed `
`"""
def _Py_ISCARRIAGERETURN(ch):
    """Check if character is carriage return `\r`"""
