from _typeshed import Incomplete
from numba import config as config

_fields: str
_gdb_info: Incomplete

class _GDBTestWrapper:
    """Wraps the gdb binary and has methods for checking what the gdb binary
    has support for (Python and NumPy)."""
    _gdb_binary: Incomplete
    def __init__(self) -> None: ...
    def _run_cmd(self, cmd=()): ...
    @property
    def gdb_binary(self): ...
    @classmethod
    def success(cls, status): ...
    def check_launch(self):
        """Checks that gdb will launch ok"""
    def check_python(self): ...
    def check_numpy(self): ...
    def check_numpy_version(self): ...

def collect_gdbinfo():
    """Prints information to stdout about the gdb setup that Numba has found"""
def display_gdbinfo(sep_pos: int = 45) -> None:
    """Displays the information collected by collect_gdbinfo.
    """
