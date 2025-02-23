from llvmlite.binding import ffi as ffi
from llvmlite.binding.common import _encode_string as _encode_string

def set_option(name, option) -> None:
    '''
    Set the given LLVM "command-line" option.

    For example set_option("test", "-debug-pass=Structure") would display
    all optimization passes when generating code.
    '''
