"""Memory management utilities.

https://github.com/pandas-dev/pandas/issues/2659#issuecomment-2452943964
"""
import ctypes
import ctypes.wintypes
import gc
import logging
import platform

logger = logging.getLogger(__name__)

def trim_windows_process_memory(pid: int | None = None) -> bool:
    """Cause effects similar to malloc_trim on -nix."""
    # Define SIZE_T based on the platform (32-bit or 64-bit)
    if ctypes.sizeof(ctypes.c_void_p) == 4:
        SIZE_T = ctypes.c_uint32
    else:
        SIZE_T = ctypes.c_uint64

    # Get a handle to the current process
    if not pid:
        pid = ctypes.windll.kernel32.GetCurrentProcess()

    # Define argument and return types for SetProcessWorkingSetSizeEx
    ctypes.windll.kernel32.SetProcessWorkingSetSizeEx.argtypes = [
        ctypes.wintypes.HANDLE,  # Process handle
        SIZE_T,  # Minimum working set size
        SIZE_T,  # Maximum working set size
        ctypes.wintypes.DWORD,  # Flags
    ]
    ctypes.windll.kernel32.SetProcessWorkingSetSizeEx.restype = ctypes.wintypes.BOOL

    # Define constants for SetProcessWorkingSetSizeEx
    QUOTA_LIMITS_HARDWS_MIN_DISABLE = 0x00000002

    # Attempt to set the working set size
    result = ctypes.windll.kernel32.SetProcessWorkingSetSizeEx(pid, SIZE_T(-1), SIZE_T(-1), QUOTA_LIMITS_HARDWS_MIN_DISABLE)

    if result == 0:
        # Retrieve the error code
        error_code = ctypes.windll.kernel32.GetLastError()
        message = f"SetProcessWorkingSetSizeEx failed with error code: {error_code}"
        logger.error(message)
        return False
    else:
        return True


def trim_ram() -> None:
    """Force python garbage collection.

    Most importantly, calls malloc_trim/SetProcessWorkingSetSizeEx, which fixes pandas/libc (?) memory leak.
    """
    gc.collect()
    if platform.system() == "Windows":
        trim_windows_process_memory()
    else:
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            logger.exception("malloc_trim attempt failed")
