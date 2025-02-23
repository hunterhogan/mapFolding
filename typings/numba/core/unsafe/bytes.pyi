from numba.core import cgutils as cgutils, types as types
from numba.core.extending import intrinsic as intrinsic

def grab_byte(typingctx, data, offset): ...
def grab_uint64_t(typingctx, data, offset): ...
def memcpy_region(typingctx, dst, dst_offset, src, src_offset, nbytes, align):
    """Copy nbytes from *(src + src_offset) to *(dst + dst_offset)"""
