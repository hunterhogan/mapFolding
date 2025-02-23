from _typeshed import Incomplete
from enum import IntEnum
from numba.core import cgutils as cgutils, config as config, types as types, typing as typing
from numba.core.errors import NumbaTypeError as NumbaTypeError, TypingError as TypingError
from numba.core.extending import intrinsic as intrinsic, lower_builtin as lower_builtin, models as models, overload as overload, overload_attribute as overload_attribute, overload_method as overload_method, register_jitable as register_jitable, register_model as register_model
from numba.core.imputils import RefType as RefType, impl_ret_borrowed as impl_ret_borrowed, iternext_impl as iternext_impl
from numba.core.types import ListType as ListType, ListTypeIterableType as ListTypeIterableType, ListTypeIteratorType as ListTypeIteratorType, NoneType as NoneType, Type as Type
from numba.cpython import listobj as listobj
from numba.typed.typedobjectutils import _as_bytes as _as_bytes, _cast as _cast, _container_get_data as _container_get_data, _container_get_meminfo as _container_get_meminfo, _get_incref_decref as _get_incref_decref, _nonoptional as _nonoptional

ll_list_type: Incomplete
ll_listiter_type: Incomplete
ll_voidptr_type: Incomplete
ll_status: Incomplete
ll_ssize_t: Incomplete
ll_bytes: Incomplete
_meminfo_listptr: Incomplete
INDEXTY: Incomplete
index_types: Incomplete
DEFAULT_ALLOCATED: int

class ListModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None: ...

class ListIterModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None: ...

class ListStatus(IntEnum):
    """Status code for other list operations.
    """
    LIST_OK = (0,)
    LIST_ERR_INDEX = -1
    LIST_ERR_NO_MEMORY = -2
    LIST_ERR_MUTATED = -3
    LIST_ERR_ITER_EXHAUSTED = -4
    LIST_ERR_IMMUTABLE = -5

class ErrorHandler:
    """ErrorHandler for calling codegen functions from this file.

    Stores the state needed to raise an exception from nopython mode.
    """
    context: Incomplete
    def __init__(self, context) -> None: ...
    def __call__(self, builder, status, msg) -> None: ...

def _check_for_none_typed(lst, method) -> None: ...
def _as_meminfo(typingctx, lstobj):
    """Returns the MemInfoPointer of a list.
    """
def _from_meminfo(typingctx, mi, listtyperef):
    """Recreate a list from a MemInfoPointer
    """
def _list_codegen_set_method_table(context, builder, lp, itemty) -> None: ...
def _list_set_method_table(typingctx, lp, itemty):
    """Wrap numba_list_set_method_table
    """
def list_is(context, builder, sig, args): ...
def _call_list_free(context, builder, ptr) -> None:
    """Call numba_list_free(ptr)
    """
def _imp_dtor(context, module):
    """Define the dtor for list
    """
def new_list(item, allocated=...):
    """Construct a new list. (Not implemented in the interpreter yet)

    Parameters
    ----------
    item: TypeRef
        Item type of the new list.
    allocated: int
        number of items to pre-allocate

    """
def _add_meminfo(context, builder, lstruct) -> None: ...
def _make_list(typingctx, itemty, ptr):
    """Make a list struct with the given *ptr*

    Parameters
    ----------
    itemty: Type
        Type of the item.
    ptr : llvm pointer value
        Points to the list object.
    """
def _list_new_codegen(context, builder, itemty, new_size, error_handler): ...
def _list_new(typingctx, itemty, allocated):
    """Wrap numba_list_new.

    Allocate a new list object with zero capacity.

    Parameters
    ----------
    itemty: Type
        Type of the items
    allocated: int
        number of items to pre-allocate

    """
def impl_new_list(item, allocated=...):
    """Creates a new list.

    Parameters
    ----------
    item: Numba type
        type of the list item.
    allocated: int
        number of items to pre-allocate

    """
def impl_len(l):
    """len(list)
    """
def _list_length(typingctx, l):
    """Wrap numba_list_length

    Returns the length of the list.
    """
def impl_allocated(l):
    """list._allocated()
    """
def _list_allocated(typingctx, l):
    """Wrap numba_list_allocated

    Returns the allocation of the list.
    """
def impl_is_mutable(l):
    """list._is_mutable()"""
def _list_is_mutable(typingctx, l):
    """Wrap numba_list_is_mutable

    Returns the state of the is_mutable member
    """
def impl_make_mutable(l):
    """list._make_mutable()"""
def impl_make_immutable(l):
    """list._make_immutable()"""
def _list_set_is_mutable(typingctx, l, is_mutable):
    """Wrap numba_list_set_mutable

    Sets the state of the is_mutable member.
    """
def _list_append(typingctx, l, item):
    """Wrap numba_list_append
    """
def impl_append(l, item): ...
def fix_index(tyctx, list_ty, index_ty): ...
def handle_index(l, index):
    """Handle index.

    If the index is negative, convert it. If the index is out of range, raise
    an IndexError.
    """
def handle_slice(l, s):
    """Handle slice.

    Convert a slice object for a given list into a range object that can be
    used to index the list. Many subtle caveats here, especially if the step is
    negative.
    """
def _gen_getitem(borrowed): ...

_list_getitem: Incomplete
_list_getitem_borrowed: Incomplete

def impl_getitem(l, index): ...
def _list_setitem(typingctx, l, index, item):
    """Wrap numba_list_setitem
    """
def impl_setitem(l, index, item): ...
def impl_pop(l, index: int = -1): ...
def _list_delitem(typingctx, l, index): ...
def _list_delete_slice(typingctx, l, start, stop, step):
    """Wrap numba_list_delete_slice
    """
def impl_delitem(l, index): ...
def impl_contains(l, item): ...
def impl_count(l, item): ...
def impl_extend(l, iterable): ...
def impl_insert(l, index, item): ...
def impl_remove(l, item): ...
def impl_clear(l): ...
def impl_reverse(l): ...
def impl_copy(l): ...
def impl_index(l, item, start: Incomplete | None = None, end: Incomplete | None = None): ...
def ol_list_sort(lst, key: Incomplete | None = None, reverse: bool = False): ...
def ol_getitem_unchecked(lst, index): ...
def ol_list_hash(lst): ...
def impl_dtype(l): ...
def _equals_helper(this, other, OP): ...
def impl_equals(this, other): ...
def impl_not_equals(this, other): ...
def compare_not_none(this, other):
    """Oldschool (python 2.x) cmp.

       if this < other return -1
       if this = other return 0
       if this > other return 1
    """
def compare_some_none(this, other, this_is_none, other_is_none):
    """Oldschool (python 2.x) cmp for None typed lists.

       if this < other return -1
       if this = other return 0
       if this > other return 1
    """
def compare_helper(this, other, accepted): ...
def impl_less_than(this, other): ...
def impl_less_than_or_equal(this, other): ...
def impl_greater_than(this, other): ...
def impl_greater_than_or_equal(this, other): ...

class ListIterInstance:
    _context: Incomplete
    _builder: Incomplete
    _iter_ty: Incomplete
    _list_ty: Incomplete
    _iter: Incomplete
    def __init__(self, context, builder, iter_type, iter_val) -> None: ...
    @classmethod
    def from_list(cls, context, builder, iter_type, list_val): ...
    @classmethod
    def _size_of_list(cls, context, builder, list_ty, ll_list): ...
    @property
    def size(self): ...
    @property
    def value(self): ...
    def getitem(self, index): ...
    @property
    def index(self): ...
    @index.setter
    def index(self, value) -> None: ...

def getiter_list(context, builder, sig, args): ...
def iternext_listiter(context, builder, sig, args, result) -> None: ...
