from .simulator_init import *
from .device_init import *
from .device_init import _auto_device as _auto_device
from numba import runtests as runtests
from numba.core import config as config
from numba.cuda.compiler import compile as compile, compile_for_current_device as compile_for_current_device, compile_ptx as compile_ptx, compile_ptx_for_current_device as compile_ptx_for_current_device

implementation: str

def test(*args, **kwargs): ...
