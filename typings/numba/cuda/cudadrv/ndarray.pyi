from numba.core.registry import cpu_target as cpu_target
from numba.cuda.cudadrv import devices as devices, driver as driver

def _calc_array_sizeof(ndim):
    """
    Use the ABI size in the CPU target
    """
def ndarray_device_allocate_data(ary):
    """
    Allocate gpu data buffer
    """
