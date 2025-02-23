from _typeshed import Incomplete
from numba import gdb as gdb, gdb_breakpoint as gdb_breakpoint, gdb_init as gdb_init
from numba.core import cgutils as cgutils, config as config, errors as errors, types as types, utils as utils
from numba.core.extending import intrinsic as intrinsic, overload as overload

_path: Incomplete
_platform: Incomplete
_unix_like: Incomplete

def _confirm_gdb(need_ptrace_attach: bool = True) -> None:
    """
    Set need_ptrace_attach to True/False to indicate whether the ptrace attach
    permission is needed for this gdb use case. Mode 0 (classic) or 1
    (restricted ptrace) is required if need_ptrace_attach is True. See:
    https://www.kernel.org/doc/Documentation/admin-guide/LSM/Yama.rst
    for details on the modes.
    """
def hook_gdb(*args): ...
def hook_gdb_init(*args): ...
def init_gdb_codegen(cgctx, builder, signature, args, const_args, do_break: bool = False) -> None: ...
def gen_gdb_impl(const_args, do_break): ...
def hook_gdb_breakpoint():
    """
    Adds the Numba break point into the source
    """
def gen_bp_impl(): ...
