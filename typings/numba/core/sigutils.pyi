from numba.core import types as types, typing as typing

def is_signature(sig):
    """
    Return whether *sig* is a potentially valid signature
    specification (for user-facing APIs).
    """
def _parse_signature_string(signature_str):
    """
    Parameters
    ----------
    signature_str : str
    """
def normalize_signature(sig):
    """
    From *sig* (a signature specification), return a ``(args, return_type)``
    tuple, where ``args`` itself is a tuple of types, and ``return_type``
    can be None if not specified.
    """
