from _typeshed import Incomplete
from enum import IntEnum
from numba import _helperlib as _helperlib
from numba.core import cgutils as cgutils, types as types, typing as typing
from numba.core.errors import LoweringError as LoweringError, NumbaTypeError as NumbaTypeError, TypingError as TypingError
from numba.core.extending import intrinsic as intrinsic, lower_builtin as lower_builtin, lower_cast as lower_cast, make_attribute_wrapper as make_attribute_wrapper, models as models, overload as overload, overload_attribute as overload_attribute, overload_method as overload_method, register_model as register_model
from numba.core.imputils import RefType as RefType, impl_ret_borrowed as impl_ret_borrowed, impl_ret_untracked as impl_ret_untracked, iternext_impl as iternext_impl
from numba.core.types import DictItemsIterableType as DictItemsIterableType, DictIteratorType as DictIteratorType, DictKeysIterableType as DictKeysIterableType, DictType as DictType, DictValuesIterableType as DictValuesIterableType, Type as Type
from numba.typed.typedobjectutils import _as_bytes as _as_bytes, _cast as _cast, _container_get_data as _container_get_data, _get_equal as _get_equal, _get_incref_decref as _get_incref_decref, _nonoptional as _nonoptional, _sentry_safe_cast_default as _sentry_safe_cast_default

ll_dict_type: Incomplete
ll_dictiter_type: Incomplete
ll_voidptr_type: Incomplete
ll_status: Incomplete
ll_ssize_t: Incomplete
ll_hash = ll_ssize_t
ll_bytes: Incomplete
_meminfo_dictptr: Incomplete

class DKIX(IntEnum):
    """Special return value of dict lookup.
    """
    EMPTY = -1

class Status(IntEnum):
    """Status code for other dict operations.
    """
    OK = 0
    OK_REPLACED = 1
    ERR_NO_MEMORY = -1
    ERR_DICT_MUTATED = -2
    ERR_ITER_EXHAUSTED = -3
    ERR_DICT_EMPTY = -4
    ERR_CMP_FAILED = -5

def new_dict(key, value, n_keys: int = 0):
    """Construct a new dict with enough space for *n_keys* without a resize.

    Parameters
    ----------
    key, value : TypeRef
        Key type and value type of the new dict.
    n_keys : int, default 0
        The number of keys to insert without needing a resize.
        A value of 0 creates a dict with minimum size.
    """

class DictModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None: ...

class DictIterModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None: ...

def _raise_if_error(context, builder, status, msg) -> None:
    """Raise an internal error depending on the value of *status*
    """
def _as_meminfo(typingctx, dctobj):
    """Returns the MemInfoPointer of a dictionary.
    """
def _from_meminfo(typingctx, mi, dicttyperef):
    """Recreate a dictionary from a MemInfoPointer
    """
def _call_dict_free(context, builder, ptr) -> None:
    """Call numba_dict_free(ptr)
    """
def _imp_dtor(context, module):
    """Define the dtor for dictionary
    """
def _dict_new_sized(typingctx, n_keys, keyty, valty):
    """Wrap numba_dict_new_sized.

    Allocate a new dictionary object with enough space to hold
    *n_keys* keys without needing a resize.

    Parameters
    ----------
    keyty, valty: Type
        Type of the key and value, respectively.
    n_keys: int
        The number of keys to insert without needing a resize.
        A value of 0 creates a dict with minimum size.
    """
def _dict_set_method_table(typingctx, dp, keyty, valty):
    """Wrap numba_dict_set_method_table
    """
def _dict_insert(typingctx, d, key, hashval, val):
    """Wrap numba_dict_insert
    """
def _dict_length(typingctx, d):
    """Wrap numba_dict_length

    Returns the length of the dictionary.
    """
def _dict_dump(typingctx, d):
    """Dump the dictionary keys and values.
    Wraps numba_dict_dump for debugging.
    """
def _dict_lookup(typingctx, d, key, hashval):
    """Wrap numba_dict_lookup

    Returns 2-tuple of (intp, ?value_type)
    """
def _dict_popitem(typingctx, d):
    """Wrap numba_dict_popitem
    """
def _dict_delitem(typingctx, d, hk, ix):
    """Wrap numba_dict_delitem
    """
def _iterator_codegen(resty):
    """The common codegen for iterator intrinsics.

    Populates the iterator struct and increfs.
    """
def _dict_items(typingctx, d):
    """Get dictionary iterator for .items()"""
def _dict_keys(typingctx, d):
    """Get dictionary iterator for .keys()"""
def _dict_values(typingctx, d):
    """Get dictionary iterator for .values()"""
def _make_dict(typingctx, keyty, valty, ptr):
    """Make a dictionary struct with the given *ptr*

    Parameters
    ----------
    keyty, valty: Type
        Type of the key and value, respectively.
    ptr : llvm pointer value
        Points to the dictionary object.
    """
def impl_new_dict(key, value, n_keys: int = 0):
    """Creates a new dictionary with *key* and *value* as the type
    of the dictionary key and value, respectively. *n_keys* is the
    number of keys to insert without requiring a resize, where a
    value of 0 creates a dictionary with minimum size.
    """
def impl_len(d):
    """len(dict)
    """
def impl_len_iters(d):
    """len(dict.keys()), len(dict.values()), len(dict.items())
    """
def impl_setitem(d, key, value): ...
def impl_get(dct, key, default: Incomplete | None = None): ...
def impl_hash(dct): ...
def impl_getitem(d, key): ...
def impl_popitem(d): ...
def impl_pop(dct, key, default: Incomplete | None = None): ...
def impl_delitem(d, k): ...
def impl_contains(d, k): ...
def impl_clear(d): ...
def impl_copy(d): ...
def impl_setdefault(dct, key, default: Incomplete | None = None): ...
def impl_items(d): ...
def impl_keys(d): ...
def impl_values(d): ...
def ol_dict_update(d, other): ...
def impl_equal(da, db): ...
def impl_not_equal(da, db): ...
def impl_iterable_getiter(context, builder, sig, args):
    """Implement iter() for .keys(), .values(), .items()
    """
def impl_dict_getiter(context, builder, sig, args):
    """Implement iter(Dict).  Semantically equivalent to dict.keys()
    """
def impl_iterator_iternext(context, builder, sig, args, result) -> None: ...
def build_map(context, builder, dict_type, item_types, items): ...
def _mixed_values_to_tuple(tyctx, d): ...
def literalstrkeydict_impl_values(d): ...
def literalstrkeydict_impl_keys(d): ...
def literalstrkeydict_impl_equals(context, builder, sig, args): ...
def literalstrkeydict_impl_get(dct, *args) -> None: ...
def literalstrkeydict_impl_copy(d): ...
def _str_items_mixed_values_to_tuple(tyctx, d): ...
def literalstrkeydict_impl_items(d): ...
def literalstrkeydict_impl_contains(d, k): ...
def literalstrkeydict_impl_len(d): ...
def literalstrkeydict_banned_impl_setitem(d, key, value) -> None: ...
def literalstrkeydict_banned_impl_delitem(d, k) -> None: ...
def literalstrkeydict_banned_impl_mutators(d, *args) -> None: ...
def cast_LiteralStrKeyDict_LiteralStrKeyDict(context, builder, fromty, toty, val): ...
def cast_DictType_DictType(context, builder, fromty, toty, val): ...
