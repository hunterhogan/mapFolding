from _typeshed import Incomplete

def get_lib_dirs():
    """
    Anaconda specific
    """

DLLNAMEMAP: Incomplete
RE_VER: str

def find_lib(libname, libdir: Incomplete | None = None, platform: Incomplete | None = None, static: bool = False): ...
def find_file(pat, libdir: Incomplete | None = None): ...
