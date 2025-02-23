from _typeshed import Incomplete
from collections.abc import Generator
from numba import _helperlib as _helperlib
from typing import NamedTuple

class Extent(NamedTuple):
    begin: Incomplete
    end: Incomplete

attempt_nocopy_reshape: Incomplete

class Dim:
    """A single dimension of the array

    Attributes
    ----------
    start:
        start offset
    stop:
        stop offset
    size:
        number of items
    stride:
        item stride
    """
    __slots__: Incomplete
    start: Incomplete
    stop: Incomplete
    size: Incomplete
    stride: Incomplete
    single: Incomplete
    def __init__(self, start, stop, size, stride, single) -> None: ...
    def __getitem__(self, item): ...
    def get_offset(self, idx): ...
    def __repr__(self) -> str: ...
    def normalize(self, base): ...
    def copy(self, start: Incomplete | None = None, stop: Incomplete | None = None, size: Incomplete | None = None, stride: Incomplete | None = None, single: Incomplete | None = None): ...
    def is_contiguous(self, itemsize): ...

def compute_index(indices, dims): ...

class Element:
    is_array: bool
    extent: Incomplete
    def __init__(self, extent) -> None: ...
    def iter_contiguous_extent(self) -> Generator[Incomplete]: ...

class Array:
    """A dummy numpy array-like object.  Consider it an array without the
    actual data, but offset from the base data pointer.

    Attributes
    ----------
    dims: tuple of Dim
        describing each dimension of the array

    ndim: int
        number of dimension

    shape: tuple of int
        size of each dimension

    strides: tuple of int
        stride of each dimension

    itemsize: int
        itemsize

    extent: (start, end)
        start and end offset containing the memory region
    """
    is_array: bool
    @classmethod
    def from_desc(cls, offset, shape, strides, itemsize): ...
    dims: Incomplete
    ndim: Incomplete
    shape: Incomplete
    strides: Incomplete
    itemsize: Incomplete
    size: Incomplete
    extent: Incomplete
    flags: Incomplete
    def __init__(self, dims, itemsize) -> None: ...
    def _compute_layout(self): ...
    def _compute_extent(self): ...
    def __repr__(self) -> str: ...
    def __getitem__(self, item): ...
    @property
    def is_c_contig(self): ...
    @property
    def is_f_contig(self): ...
    def iter_contiguous_extent(self) -> Generator[Incomplete]:
        """ Generates extents
        """
    def reshape(self, *newdims, **kws): ...
    def squeeze(self, axis: Incomplete | None = None): ...
    def ravel(self, order: str = 'C'): ...

def iter_strides_f_contig(arr, shape: Incomplete | None = None) -> Generator[Incomplete]:
    """yields the f-contiguous strides
    """
def iter_strides_c_contig(arr, shape: Incomplete | None = None) -> Generator[Incomplete]:
    """yields the c-contiguous strides
    """
def is_element_indexing(item, ndim): ...
def _compute_size(start, stop, step):
    """Algorithm adapted from cpython rangeobject.c
    """
