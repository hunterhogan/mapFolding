from _typeshed import Incomplete
from collections.abc import Generator
from functools import cached_property as cached_property
from numba.core import cgutils as cgutils, types as types, typing as typing
from numba.core.errors import NumbaValueError as NumbaValueError, TypingError as TypingError
from numba.core.extending import intrinsic as intrinsic, overload as overload, overload_method as overload_method
from numba.core.imputils import RefType as RefType, call_len as call_len, for_iter as for_iter, impl_ret_borrowed as impl_ret_borrowed, impl_ret_new_ref as impl_ret_new_ref, impl_ret_untracked as impl_ret_untracked, iternext_impl as iternext_impl, lower_builtin as lower_builtin, lower_cast as lower_cast
from numba.cpython import slicing as slicing
from numba.misc import quicksort as quicksort
from typing import NamedTuple

def get_payload_struct(context, builder, set_type, ptr):
    """
    Given a set value and type, get its payload structure (as a
    reference, so that mutations are seen by all).
    """
def get_entry_size(context, set_type):
    """
    Return the entry size for the given set type.
    """

EMPTY: int
DELETED: int
FALLBACK: int
MINSIZE: int
LINEAR_PROBES: int
DEBUG_ALLOCS: bool

def get_hash_value(context, builder, typ, value):
    """
    Compute the hash of the given value.
    """
def _get_hash_value_intrinsic(typingctx, value): ...
def is_hash_empty(context, builder, h):
    """
    Whether the hash value denotes an empty entry.
    """
def is_hash_deleted(context, builder, h):
    """
    Whether the hash value denotes a deleted entry.
    """
def is_hash_used(context, builder, h):
    """
    Whether the hash value denotes an active entry.
    """
def check_all_set(*args) -> None: ...

class SetLoop(NamedTuple):
    index: Incomplete
    entry: Incomplete
    do_break: Incomplete

class _SetPayload:
    _context: Incomplete
    _builder: Incomplete
    _ty: Incomplete
    _payload: Incomplete
    _entries: Incomplete
    _ptr: Incomplete
    def __init__(self, context, builder, set_type, ptr) -> None: ...
    @property
    def mask(self): ...
    @mask.setter
    def mask(self, value) -> None: ...
    @property
    def used(self): ...
    @used.setter
    def used(self, value) -> None: ...
    @property
    def fill(self): ...
    @fill.setter
    def fill(self, value) -> None: ...
    @property
    def finger(self): ...
    @finger.setter
    def finger(self, value) -> None: ...
    @property
    def dirty(self): ...
    @dirty.setter
    def dirty(self, value) -> None: ...
    @property
    def entries(self):
        """
        A pointer to the start of the entries array.
        """
    @property
    def ptr(self):
        """
        A pointer to the start of the NRT-allocated area.
        """
    def get_entry(self, idx):
        """
        Get entry number *idx*.
        """
    def _lookup(self, item, h, for_insert: bool = False):
        """
        Lookup the *item* with the given hash values in the entries.

        Return a (found, entry index) tuple:
        - If found is true, <entry index> points to the entry containing
          the item.
        - If found is false, <entry index> points to the empty entry that
          the item can be written to (only if *for_insert* is true)
        """
    def _iterate(self, start: Incomplete | None = None) -> Generator[Incomplete]:
        """
        Iterate over the payload's entries.  Yield a SetLoop.
        """
    def _next_entry(self) -> Generator[Incomplete]:
        """
        Yield a random entry from the payload.  Caller must ensure the
        set isn't empty, otherwise the function won't end.
        """

class SetInstance:
    _context: Incomplete
    _builder: Incomplete
    _ty: Incomplete
    _entrysize: Incomplete
    _set: Incomplete
    def __init__(self, context, builder, set_type, set_val) -> None: ...
    @property
    def dtype(self): ...
    @property
    def payload(self):
        """
        The _SetPayload for this set.
        """
    @property
    def value(self): ...
    @property
    def meminfo(self): ...
    @property
    def parent(self): ...
    @parent.setter
    def parent(self, value) -> None: ...
    def get_size(self):
        """
        Return the number of elements in the size.
        """
    def set_dirty(self, val) -> None: ...
    def _add_entry(self, payload, entry, item, h, do_resize: bool = True) -> None: ...
    def _add_key(self, payload, item, h, do_resize: bool = True, do_incref: bool = True) -> None: ...
    def _remove_entry(self, payload, entry, do_resize: bool = True, do_decref: bool = True) -> None: ...
    def _remove_key(self, payload, item, h, do_resize: bool = True): ...
    def add(self, item, do_resize: bool = True) -> None: ...
    def add_pyapi(self, pyapi, item, do_resize: bool = True) -> None:
        """A version of .add for use inside functions following Python calling
        convention.
        """
    def _pyapi_get_hash_value(self, pyapi, context, builder, item):
        """Python API compatible version of `get_hash_value()`.
        """
    def contains(self, item): ...
    def discard(self, item): ...
    def pop(self): ...
    def clear(self) -> None: ...
    def copy(self):
        """
        Return a copy of this set.
        """
    def intersect(self, other) -> None:
        """
        In-place intersection with *other* set.
        """
    def difference(self, other) -> None:
        """
        In-place difference with *other* set.
        """
    def symmetric_difference(self, other) -> None:
        """
        In-place symmetric difference with *other* set.
        """
    def issubset(self, other, strict: bool = False): ...
    def isdisjoint(self, other): ...
    def equals(self, other): ...
    @classmethod
    def allocate_ex(cls, context, builder, set_type, nitems: Incomplete | None = None):
        """
        Allocate a SetInstance with its storage.
        Return a (ok, instance) tuple where *ok* is a LLVM boolean and
        *instance* is a SetInstance object (the object's contents are
        only valid when *ok* is true).
        """
    @classmethod
    def allocate(cls, context, builder, set_type, nitems: Incomplete | None = None):
        """
        Allocate a SetInstance with its storage.  Same as allocate_ex(),
        but return an initialized *instance*.  If allocation failed,
        control is transferred to the caller using the target's current
        call convention.
        """
    @classmethod
    def from_meminfo(cls, context, builder, set_type, meminfo):
        """
        Allocate a new set instance pointing to an existing payload
        (a meminfo pointer).
        Note the parent field has to be filled by the caller.
        """
    @classmethod
    def choose_alloc_size(cls, context, builder, nitems):
        """
        Choose a suitable number of entries for the given number of items.
        """
    def upsize(self, nitems) -> None:
        """
        When adding to the set, ensure it is properly sized for the given
        number of used entries.
        """
    def downsize(self, nitems) -> None:
        """
        When removing from the set, ensure it is properly sized for the given
        number of used entries.
        """
    def _resize(self, payload, nentries, errmsg) -> None:
        """
        Resize the payload to the given number of entries.

        CAUTION: *nentries* must be a power of 2!
        """
    def _replace_payload(self, nentries) -> None:
        """
        Replace the payload with a new empty payload with the given number
        of entries.

        CAUTION: *nentries* must be a power of 2!
        """
    def _allocate_payload(self, nentries, realloc: bool = False):
        """
        Allocate and initialize payload for the given number of entries.
        If *realloc* is True, the existing meminfo is reused.

        CAUTION: *nentries* must be a power of 2!
        """
    def _free_payload(self, ptr) -> None:
        """
        Free an allocated old payload at *ptr*.
        """
    def _copy_payload(self, src_payload):
        """
        Raw-copy the given payload into self.
        """
    def _imp_dtor(self, context, module):
        """Define the dtor for set
        """
    def incref_value(self, val) -> None:
        """Incref an element value
        """
    def decref_value(self, val) -> None:
        """Decref an element value
        """

class SetIterInstance:
    _context: Incomplete
    _builder: Incomplete
    _ty: Incomplete
    _iter: Incomplete
    _payload: Incomplete
    def __init__(self, context, builder, iter_type, iter_val) -> None: ...
    @classmethod
    def from_set(cls, context, builder, iter_type, set_val): ...
    @property
    def value(self): ...
    @property
    def meminfo(self): ...
    @property
    def index(self): ...
    @index.setter
    def index(self, value) -> None: ...
    def iternext(self, result) -> None: ...

def build_set(context, builder, set_type, items):
    """
    Build a set of the given type, containing the given items.
    """
def set_empty_constructor(context, builder, sig, args): ...
def set_constructor(context, builder, sig, args): ...
def set_len(context, builder, sig, args): ...
def in_set(context, builder, sig, args): ...
def getiter_set(context, builder, sig, args): ...
def iternext_listiter(context, builder, sig, args, result) -> None: ...
def set_add(context, builder, sig, args): ...
def _set_discard(typingctx, s, item): ...
def ol_set_discard(s, item): ...
def _set_pop(typingctx, s): ...
def ol_set_pop(s): ...
def _set_remove(typingctx, s, item): ...
def ol_set_remove(s, item): ...
def _set_clear(typingctx, s): ...
def ol_set_clear(s): ...
def _set_copy(typingctx, s): ...
def ol_set_copy(s): ...
def set_difference_update(context, builder, sig, args): ...
def _set_difference_update(typingctx, a, b): ...
def set_difference_update_impl(a, b): ...
def set_intersection_update(context, builder, sig, args): ...
def _set_intersection_update(typingctx, a, b): ...
def set_intersection_update_impl(a, b): ...
def set_symmetric_difference_update(context, builder, sig, args): ...
def _set_symmetric_difference_update(typingctx, a, b): ...
def set_symmetric_difference_update_impl(a, b): ...
def set_update(context, builder, sig, args): ...
def gen_operator_impl(op, impl): ...
def impl_set_difference(a, b): ...
def set_intersection(a, b): ...
def set_symmetric_difference(a, b): ...
def set_union(a, b): ...
def _set_isdisjoint(typingctx, a, b): ...
def set_isdisjoint(a, b): ...
def _set_issubset(typingctx, a, b): ...
def set_issubset(a, b): ...
def set_issuperset(a, b): ...
def _set_eq(typingctx, a, b): ...
def set_eq(a, b): ...
def set_ne(a, b): ...
def _set_lt(typingctx, a, b): ...
def set_lt(a, b): ...
def set_gt(a, b): ...
def set_is(context, builder, sig, args): ...
def set_to_set(context, builder, fromty, toty, val): ...
