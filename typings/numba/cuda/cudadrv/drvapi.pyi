from _typeshed import Incomplete
from ctypes import c_int, c_size_t, c_void_p
from numba.cuda.cudadrv import _extras as _extras

cu_device = c_int
cu_device_attribute = c_int
cu_context = c_void_p
cu_module = c_void_p
cu_jit_option = c_int
cu_jit_input_type = c_int
cu_function = c_void_p
cu_device_ptr = c_size_t
cu_stream = c_void_p
cu_event = c_void_p
cu_link_state = c_void_p
cu_function_attribute = c_int
cu_ipc_mem_handle: Incomplete
cu_uuid: Incomplete
cu_stream_callback_pyobj: Incomplete
cu_occupancy_b2d_size: Incomplete
CU_STREAM_DEFAULT: int
CU_STREAM_LEGACY: int
CU_STREAM_PER_THREAD: int
API_PROTOTYPES: Incomplete
