from llvmlite.binding import ffi as ffi
from llvmlite.binding.common import _encode_string as _encode_string

def address_of_symbol(name):
    """
    Get the in-process address of symbol named *name*.
    An integer is returned, or None if the symbol isn't found.
    """
def add_symbol(name, address) -> None:
    """
    Register the *address* of global symbol *name*.  This will make
    it usable (e.g. callable) from LLVM-compiled functions.
    """
def load_library_permanently(filename) -> None:
    """
    Load an external library
    """
