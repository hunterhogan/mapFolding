from .api import *
from . import compiler as compiler
from .cudadrv import devicearray as devicearray
from .cudadrv.devicearray import auto_device as auto_device, device_array as device_array, device_array_like as device_array_like, mapped_array as mapped_array, pinned as pinned, pinned_array as pinned_array, pinned_array_like as pinned_array_like, to_device as to_device
from .cudadrv.devices import gpus as gpus, require_context as require_context
from .cudadrv.runtime import runtime as runtime
from .reduction import Reduce as Reduce
from .vector_types import vector_types as vector_types
from numba.core import config as config
from numba.cuda.simulator import cudadrv as cudadrv

reduce = Reduce
