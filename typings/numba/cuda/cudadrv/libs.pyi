from numba.cuda.cuda_paths import get_cuda_paths as get_cuda_paths
from numba.cuda.cudadrv.driver import load_driver as load_driver, locate_driver_and_loader as locate_driver_and_loader
from numba.cuda.cudadrv.error import CudaSupportError as CudaSupportError
from numba.misc.findlib import find_lib as find_lib

_dllnamepattern: str
_staticnamepattern: str

def get_libdevice(): ...
def open_libdevice(): ...
def get_cudalib(lib, static: bool = False):
    """
    Find the path of a CUDA library based on a search of known locations. If
    the search fails, return a generic filename for the library (e.g.
    'libnvvm.so' for 'nvvm') so that we may attempt to load it using the system
    loader's search mechanism.
    """
def open_cudalib(lib): ...
def check_static_lib(path) -> None: ...
def _get_source_variable(lib, static: bool = False): ...
def test():
    """Test library lookup.  Path info is printed to stdout.
    """
