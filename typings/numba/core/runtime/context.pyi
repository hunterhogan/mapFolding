from _typeshed import Incomplete
from numba.core import cgutils as cgutils, config as config, errors as errors, types as types
from numba.core.utils import PYVERSION as PYVERSION
from typing import NamedTuple

class _NRT_Meminfo_Functions(NamedTuple):
    alloc: Incomplete
    alloc_dtor: Incomplete
    alloc_aligned: Incomplete

_NRT_MEMINFO_SAFE_API: Incomplete
_NRT_MEMINFO_DEFAULT_API: Incomplete

class NRTContext:
    """
    An object providing access to NRT APIs in the lowering pass.
    """
    _context: Incomplete
    _enabled: Incomplete
    _meminfo_api: Incomplete
    def __init__(self, context, enabled) -> None: ...
    def _require_nrt(self) -> None: ...
    def _check_null_result(func): ...
    def allocate(self, builder, size):
        """
        Low-level allocate a new memory area of `size` bytes. The result of the
        call is checked and if it is NULL, i.e. allocation failed, then a
        MemoryError is raised.
        """
    def allocate_unchecked(self, builder, size):
        """
        Low-level allocate a new memory area of `size` bytes. Returns NULL to
        indicate error/failure to allocate.
        """
    def free(self, builder, ptr):
        """
        Low-level free a memory area allocated with allocate().
        """
    def meminfo_alloc(self, builder, size):
        """
        Allocate a new MemInfo with a data payload of `size` bytes.

        A pointer to the MemInfo is returned.

        The result of the call is checked and if it is NULL, i.e. allocation
        failed, then a MemoryError is raised.
        """
    def meminfo_alloc_unchecked(self, builder, size):
        """
        Allocate a new MemInfo with a data payload of `size` bytes.

        A pointer to the MemInfo is returned.

        Returns NULL to indicate error/failure to allocate.
        """
    def meminfo_alloc_dtor(self, builder, size, dtor):
        """
        Allocate a new MemInfo with a data payload of `size` bytes and a
        destructor `dtor`.

        A pointer to the MemInfo is returned.

        The result of the call is checked and if it is NULL, i.e. allocation
        failed, then a MemoryError is raised.
        """
    def meminfo_alloc_dtor_unchecked(self, builder, size, dtor):
        """
        Allocate a new MemInfo with a data payload of `size` bytes and a
        destructor `dtor`.

        A pointer to the MemInfo is returned.

        Returns NULL to indicate error/failure to allocate.
        """
    def meminfo_alloc_aligned(self, builder, size, align):
        """
        Allocate a new MemInfo with an aligned data payload of `size` bytes.
        The data pointer is aligned to `align` bytes.  `align` can be either
        a Python int or a LLVM uint32 value.

        A pointer to the MemInfo is returned.

        The result of the call is checked and if it is NULL, i.e. allocation
        failed, then a MemoryError is raised.
        """
    def meminfo_alloc_aligned_unchecked(self, builder, size, align):
        """
        Allocate a new MemInfo with an aligned data payload of `size` bytes.
        The data pointer is aligned to `align` bytes.  `align` can be either
        a Python int or a LLVM uint32 value.

        A pointer to the MemInfo is returned.

        Returns NULL to indicate error/failure to allocate.
        """
    def meminfo_new_varsize(self, builder, size):
        """
        Allocate a MemInfo pointing to a variable-sized data area.  The area
        is separately allocated (i.e. two allocations are made) so that
        re-allocating it doesn't change the MemInfo's address.

        A pointer to the MemInfo is returned.

        The result of the call is checked and if it is NULL, i.e. allocation
        failed, then a MemoryError is raised.
        """
    def meminfo_new_varsize_unchecked(self, builder, size):
        """
        Allocate a MemInfo pointing to a variable-sized data area.  The area
        is separately allocated (i.e. two allocations are made) so that
        re-allocating it doesn't change the MemInfo's address.

        A pointer to the MemInfo is returned.

        Returns NULL to indicate error/failure to allocate.
        """
    def meminfo_new_varsize_dtor(self, builder, size, dtor):
        """
        Like meminfo_new_varsize() but also set the destructor for
        cleaning up references to objects inside the allocation.

        A pointer to the MemInfo is returned.

        The result of the call is checked and if it is NULL, i.e. allocation
        failed, then a MemoryError is raised.
        """
    def meminfo_new_varsize_dtor_unchecked(self, builder, size, dtor):
        """
        Like meminfo_new_varsize() but also set the destructor for
        cleaning up references to objects inside the allocation.

        A pointer to the MemInfo is returned.

        Returns NULL to indicate error/failure to allocate.
        """
    def meminfo_varsize_alloc(self, builder, meminfo, size):
        """
        Allocate a new data area for a MemInfo created by meminfo_new_varsize().
        The new data pointer is returned, for convenience.

        Contrary to realloc(), this always allocates a new area and doesn't
        copy the old data.  This is useful if resizing a container needs
        more than simply copying the data area (e.g. for hash tables).

        The old pointer will have to be freed with meminfo_varsize_free().

        The result of the call is checked and if it is NULL, i.e. allocation
        failed, then a MemoryError is raised.
        """
    def meminfo_varsize_alloc_unchecked(self, builder, meminfo, size):
        """
        Allocate a new data area for a MemInfo created by meminfo_new_varsize().
        The new data pointer is returned, for convenience.

        Contrary to realloc(), this always allocates a new area and doesn't
        copy the old data.  This is useful if resizing a container needs
        more than simply copying the data area (e.g. for hash tables).

        The old pointer will have to be freed with meminfo_varsize_free().

        Returns NULL to indicate error/failure to allocate.
        """
    def meminfo_varsize_realloc(self, builder, meminfo, size):
        """
        Reallocate a data area allocated by meminfo_new_varsize().
        The new data pointer is returned, for convenience.

        The result of the call is checked and if it is NULL, i.e. allocation
        failed, then a MemoryError is raised.
        """
    def meminfo_varsize_realloc_unchecked(self, builder, meminfo, size):
        """
        Reallocate a data area allocated by meminfo_new_varsize().
        The new data pointer is returned, for convenience.

        Returns NULL to indicate error/failure to allocate.
        """
    def meminfo_varsize_free(self, builder, meminfo, ptr):
        """
        Free a memory area allocated for a NRT varsize object.
        Note this does *not* free the NRT object itself!
        """
    def _call_varsize_alloc(self, builder, meminfo, size, funcname): ...
    def meminfo_data(self, builder, meminfo):
        """
        Given a MemInfo pointer, return a pointer to the allocated data
        managed by it.  This works for MemInfos allocated with all the
        above methods.
        """
    def get_meminfos(self, builder, ty, val):
        """Return a list of *(type, meminfo)* inside the given value.
        """
    def _call_incref_decref(self, builder, typ, value, funcname) -> None:
        """Call function of *funcname* on every meminfo found in *value*.
        """
    def incref(self, builder, typ, value) -> None:
        """
        Recursively incref the given *value* and its members.
        """
    def decref(self, builder, typ, value) -> None:
        """
        Recursively decref the given *value* and its members.
        """
    def get_nrt_api(self, builder):
        """Calls NRT_get_api(), which returns the NRT API function table.
        """
    def eh_check(self, builder):
        """Check if an exception is raised
        """
    def eh_try(self, builder) -> None:
        """Begin a try-block.
        """
    def eh_end_try(self, builder) -> None:
        """End a try-block
        """
