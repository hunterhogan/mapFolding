from _typeshed import Incomplete
from numba.core import config as config, types as types

_re_invalid_char: Incomplete
PREFIX: str
N2CODE: Incomplete

def _escape_string(text):
    '''Escape the given string so that it only contains ASCII characters
    of [a-zA-Z0-9_$].

    The dollar symbol ($) and other invalid characters are escaped into
    the string sequence of "$xx" where "xx" is the hex codepoint of the char.

    Multibyte characters are encoded into utf8 and converted into the above
    hex format.
    '''
def _fix_lead_digit(text):
    """
    Fix text with leading digit
    """
def _len_encoded(string):
    """
    Prefix string with digit indicating the length.
    Add underscore if string is prefixed with digits.
    """
def mangle_abi_tag(abi_tag: str) -> str: ...
def mangle_identifier(ident, template_params: str = '', *, abi_tags=(), uid: Incomplete | None = None):
    """
    Mangle the identifier with optional template parameters and abi_tags.

    Note:

    This treats '.' as '::' in C++.
    """
def mangle_type_or_value(typ):
    """
    Mangle type parameter and arbitrary value.
    """
mangle_type = mangle_type_or_value
mangle_value = mangle_type_or_value

def mangle_templated_ident(identifier, parameters):
    """
    Mangle templated identifier.
    """
def mangle_args(argtys):
    """
    Mangle sequence of Numba type objects and arbitrary values.
    """
def mangle(ident, argtys, *, abi_tags=(), uid: Incomplete | None = None):
    """
    Mangle identifier with Numba type objects and abi-tags.
    """
def prepend_namespace(mangled, ns):
    """
    Prepend namespace to mangled name.
    """
def _split_mangled_ident(mangled):
    """
    Returns `(head, tail)` where `head` is the `<len> + <name>` encoded
    identifier and `tail` is the remaining.
    """
