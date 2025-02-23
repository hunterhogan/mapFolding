from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import ExitStack
from numba.core import config as config, debuginfo as debuginfo, types as types, utils as utils
from typing import NamedTuple

bool_t: Incomplete
int8_t: Incomplete
int32_t: Incomplete
intp_t: Incomplete
voidptr_t: Incomplete
true_bit: Incomplete
false_bit: Incomplete
true_byte: Incomplete
false_byte: Incomplete

def as_bool_bit(builder, value): ...
def make_anonymous_struct(builder, values, struct_type: Incomplete | None = None):
    """
    Create an anonymous struct containing the given LLVM *values*.
    """
def make_bytearray(buf):
    """
    Make a byte array constant from *buf*.
    """

_struct_proxy_cache: Incomplete

def create_struct_proxy(fe_type, kind: str = 'value'):
    """
    Returns a specialized StructProxy subclass for the given fe_type.
    """
def copy_struct(dst, src, repl={}):
    """
    Copy structure from *src* to *dst* with replacement from *repl*.
    """

class _StructProxy:
    """
    Creates a `Structure` like interface that is constructed with information
    from DataModel instance.  FE type must have a data model that is a
    subclass of StructModel.
    """
    _fe_type: Incomplete
    _context: Incomplete
    _datamodel: Incomplete
    _builder: Incomplete
    _be_type: Incomplete
    _value: Incomplete
    _outer_ref: Incomplete
    def __init__(self, context, builder, value: Incomplete | None = None, ref: Incomplete | None = None) -> None: ...
    def _make_refs(self, ref):
        """
        Return an (outer ref, value ref) pair.  By default, these are
        the same pointers, but a derived class may override this.
        """
    def _get_be_type(self, datamodel) -> None: ...
    def _cast_member_to_value(self, index, val) -> None: ...
    def _cast_member_from_value(self, index, val) -> None: ...
    def _get_ptr_by_index(self, index): ...
    def _get_ptr_by_name(self, attrname): ...
    def __getattr__(self, field):
        """
        Load the LLVM value of the named *field*.
        """
    def __setattr__(self, field, value):
        """
        Store the LLVM *value* into the named *field*.
        """
    def __getitem__(self, index):
        """
        Load the LLVM value of the field at *index*.
        """
    def __setitem__(self, index, value) -> None:
        """
        Store the LLVM *value* into the field at *index*.
        """
    def __len__(self) -> int:
        """
        Return the number of fields.
        """
    def _getpointer(self):
        """
        Return the LLVM pointer to the underlying structure.
        """
    def _getvalue(self):
        """
        Load and return the value of the underlying LLVM structure.
        """
    def _setvalue(self, value) -> None:
        """
        Store the value in this structure.
        """

class ValueStructProxy(_StructProxy):
    """
    Create a StructProxy suitable for accessing regular values
    (e.g. LLVM values or alloca slots).
    """
    def _get_be_type(self, datamodel): ...
    def _cast_member_to_value(self, index, val): ...
    def _cast_member_from_value(self, index, val): ...

class DataStructProxy(_StructProxy):
    """
    Create a StructProxy suitable for accessing data persisted in memory.
    """
    def _get_be_type(self, datamodel): ...
    def _cast_member_to_value(self, index, val): ...
    def _cast_member_from_value(self, index, val): ...

class Structure:
    """
    A high-level object wrapping a alloca'ed LLVM structure, including
    named fields and attribute access.
    """
    _type: Incomplete
    _context: Incomplete
    _builder: Incomplete
    _value: Incomplete
    _namemap: Incomplete
    _fdmap: Incomplete
    _typemap: Incomplete
    def __init__(self, context, builder, value: Incomplete | None = None, ref: Incomplete | None = None, cast_ref: bool = False) -> None: ...
    def _get_ptr_by_index(self, index): ...
    def _get_ptr_by_name(self, attrname): ...
    def __getattr__(self, field):
        """
        Load the LLVM value of the named *field*.
        """
    def __setattr__(self, field, value):
        """
        Store the LLVM *value* into the named *field*.
        """
    def __getitem__(self, index):
        """
        Load the LLVM value of the field at *index*.
        """
    def __setitem__(self, index, value) -> None:
        """
        Store the LLVM *value* into the field at *index*.
        """
    def __len__(self) -> int:
        """
        Return the number of fields.
        """
    def _getpointer(self):
        """
        Return the LLVM pointer to the underlying structure.
        """
    def _getvalue(self):
        """
        Load and return the value of the underlying LLVM structure.
        """
    def _setvalue(self, value) -> None:
        """Store the value in this structure"""

def alloca_once(builder, ty, size: Incomplete | None = None, name: str = '', zfill: bool = False):
    """Allocate stack memory at the entry block of the current function
    pointed by ``builder`` with llvm type ``ty``.  The optional ``size`` arg
    set the number of element to allocate.  The default is 1.  The optional
    ``name`` arg set the symbol name inside the llvm IR for debugging.
    If ``zfill`` is set, fill the memory with zeros at the current
    use-site location.  Note that the memory is always zero-filled after the
    ``alloca`` at init-site (the entry block).
    """
def sizeof(builder, ptr_type):
    """Compute sizeof using GEP
    """
def alloca_once_value(builder, value, name: str = '', zfill: bool = False):
    """
    Like alloca_once(), but passing a *value* instead of a type.  The
    type is inferred and the allocated slot is also initialized with the
    given value.
    """
def insert_pure_function(module, fnty, name):
    """
    Insert a pure function (in the functional programming sense) in the
    given module.
    """
def get_or_insert_function(module, fnty, name):
    """
    Get the function named *name* with type *fnty* from *module*, or insert it
    if it doesn't exist.
    """
def get_or_insert_named_metadata(module, name): ...
def add_global_variable(module, ty, name, addrspace: int = 0): ...
def terminate(builder, bbend) -> None: ...
def get_null_value(ltype): ...
def is_null(builder, val): ...
def is_not_null(builder, val): ...
def if_unlikely(builder, pred): ...
def if_likely(builder, pred): ...
def ifnot(builder, pred): ...
def increment_index(builder, val):
    """
    Increment an index *val*.
    """

class Loop(NamedTuple):
    index: Incomplete
    do_break: Incomplete

def for_range(builder, count, start: Incomplete | None = None, intp: Incomplete | None = None) -> Generator[Incomplete]:
    """
    Generate LLVM IR for a for-loop in [start, count).
    *start* is equal to 0 by default.

    Yields a Loop namedtuple with the following members:
    - `index` is the loop index's value
    - `do_break` is a no-argument callable to break out of the loop
    """
def for_range_slice(builder, start, stop, step, intp: Incomplete | None = None, inc: bool = True) -> Generator[Incomplete]:
    """
    Generate LLVM IR for a for-loop based on a slice.  Yields a
    (index, count) tuple where `index` is the slice index's value
    inside the loop, and `count` the iteration count.

    Parameters
    -------------
    builder : object
        IRBuilder object
    start : int
        The beginning value of the slice
    stop : int
        The end value of the slice
    step : int
        The step value of the slice
    intp :
        The data type
    inc : boolean, optional
        Signals whether the step is positive (True) or negative (False).

    Returns
    -----------
        None
    """
def for_range_slice_generic(builder, start, stop, step) -> Generator[Incomplete]:
    """
    A helper wrapper for for_range_slice().  This is a context manager which
    yields two for_range_slice()-alike context managers, the first for
    the positive step case, the second for the negative step case.

    Use:
        with for_range_slice_generic(...) as (pos_range, neg_range):
            with pos_range as (idx, count):
                ...
            with neg_range as (idx, count):
                ...
    """
def loop_nest(builder, shape, intp, order: str = 'C') -> Generator[Incomplete, None, Incomplete]:
    """
    Generate a loop nest walking a N-dimensional array.
    Yields a tuple of N indices for use in the inner loop body,
    iterating over the *shape* space.

    If *order* is 'C' (the default), indices are incremented inside-out
    (i.e. (0,0), (0,1), (0,2), (1,0) etc.).
    If *order* is 'F', they are incremented outside-in
    (i.e. (0,0), (1,0), (2,0), (0,1) etc.).
    This has performance implications when walking an array as it impacts
    the spatial locality of memory accesses.
    """
def _loop_nest(builder, shape, intp) -> Generator[Incomplete]: ...
def pack_array(builder, values, ty: Incomplete | None = None):
    """
    Pack a sequence of values in a LLVM array.  *ty* should be given
    if the array may be empty, in which case the type can't be inferred
    from the values.
    """
def pack_struct(builder, values):
    """
    Pack a sequence of values into a LLVM struct.
    """
def unpack_tuple(builder, tup, count: Incomplete | None = None):
    """
    Unpack an array or structure of values, return a Python tuple.
    """
def get_item_pointer(context, builder, aryty, ary, inds, wraparound: bool = False, boundscheck: bool = False): ...
def do_boundscheck(context, builder, ind, dimlen, axis: Incomplete | None = None) -> None: ...
def get_item_pointer2(context, builder, data, shape, strides, layout, inds, wraparound: bool = False, boundscheck: bool = False): ...
def _scalar_pred_against_zero(builder, value, fpred, icond): ...
def is_scalar_zero(builder, value):
    """
    Return a predicate representing whether *value* is equal to zero.
    """
def is_not_scalar_zero(builder, value):
    '''
    Return a predicate representing whether a *value* is not equal to zero.
    (not exactly "not is_scalar_zero" because of nans)
    '''
def is_scalar_zero_or_nan(builder, value):
    """
    Return a predicate representing whether *value* is equal to either zero
    or NaN.
    """
is_true = is_not_scalar_zero
is_false = is_scalar_zero

def is_scalar_neg(builder, value):
    """
    Is *value* negative?  Assumes *value* is signed.
    """
def early_exit_if(builder, stack: ExitStack, cond):
    """
    The Python code::

        with contextlib.ExitStack() as stack:
            with early_exit_if(builder, stack, cond):
                cleanup()
            body()

    emits the code::

        if (cond) {
            <cleanup>
        }
        else {
            <body>
        }

    This can be useful for generating code with lots of early exits, without
    having to increase the indentation each time.
    """
def early_exit_if_null(builder, stack, obj):
    """
    A convenience wrapper for :func:`early_exit_if`, for the common case where
    the CPython API indicates an error by returning ``NULL``.
    """
def guard_null(context, builder, value, exc_tuple) -> None:
    """
    Guard against *value* being null or zero.
    *exc_tuple* should be a (exception type, arguments...) tuple.
    """
def guard_memory_error(context, builder, pointer, msg: Incomplete | None = None) -> None:
    """
    Guard against *pointer* being NULL (and raise a MemoryError).
    """
def if_zero(builder, value, likely: bool = False) -> Generator[None]:
    """
    Execute the given block if the scalar value is zero.
    """
guard_zero = guard_null

def is_pointer(ltyp):
    """
    Whether the LLVM type *typ* is a struct type.
    """
def get_record_member(builder, record, offset, typ): ...
def is_neg_int(builder, val): ...
def gep_inbounds(builder, ptr, *inds, **kws):
    """
    Same as *gep*, but add the `inbounds` keyword.
    """
def gep(builder, ptr, *inds, **kws):
    """
    Emit a getelementptr instruction for the given pointer and indices.
    The indices can be LLVM values or Python int constants.
    """
def pointer_add(builder, ptr, offset, return_type: Incomplete | None = None):
    """
    Add an integral *offset* to pointer *ptr*, and return a pointer
    of *return_type* (or, if omitted, the same type as *ptr*).

    Note the computation is done in bytes, and ignores the width of
    the pointed item type.
    """
def memset(builder, ptr, size, value) -> None:
    """
    Fill *size* bytes starting from *ptr* with *value*.
    """
def memset_padding(builder, ptr) -> None:
    """
    Fill padding bytes of the pointee with zeros.
    """
def global_constant(builder_or_module, name, value, linkage: str = 'internal'):
    """
    Get or create a (LLVM module-)global constant with *name* or *value*.
    """
def divmod_by_constant(builder, val, divisor):
    """
    Compute the (quotient, remainder) of *val* divided by the constant
    positive *divisor*.  The semantics reflects those of Python integer
    floor division, rather than C's / LLVM's signed division and modulo.
    The difference lies with a negative *val*.
    """
def cbranch_or_continue(builder, cond, bbtrue):
    """
    Branch conditionally or continue.

    Note: a new block is created and builder is moved to the end of the new
          block.
    """
def memcpy(builder, dst, src, count) -> None:
    """
    Emit a memcpy to the builder.

    Copies each element of dst to src. Unlike the C equivalent, each element
    can be any LLVM type.

    Assumes
    -------
    * dst.type == src.type
    * count is positive
    """
def _raw_memcpy(builder, func_name, dst, src, count, itemsize, align) -> None: ...
def raw_memcpy(builder, dst, src, count, itemsize, align: int = 1):
    """
    Emit a raw memcpy() call for `count` items of size `itemsize`
    from `src` to `dest`.
    """
def raw_memmove(builder, dst, src, count, itemsize, align: int = 1):
    """
    Emit a raw memmove() call for `count` items of size `itemsize`
    from `src` to `dest`.
    """
def muladd_with_overflow(builder, a, b, c):
    """
    Compute (a * b + c) and return a (result, overflow bit) pair.
    The operands must be signed integers.
    """
def printf(builder, format, *args):
    """
    Calls printf().
    Argument `format` is expected to be a Python string.
    Values to be printed are listed in `args`.

    Note: There is no checking to ensure there is correct number of values
    in `args` and there type matches the declaration in the format string.
    """
def snprintf(builder, buffer, bufsz, format, *args):
    """Calls libc snprintf(buffer, bufsz, format, ...args)
    """
def snprintf_stackbuffer(builder, bufsz, format, *args):
    """Similar to `snprintf()` but the buffer is stack allocated to size
    *bufsz*.

    Returns the buffer pointer as i8*.
    """
def normalize_ir_text(text):
    """
    Normalize the given string to latin1 compatible encoding that is
    suitable for use in LLVM IR.
    """
def hexdump(builder, ptr, nbytes) -> None:
    """Debug print the memory region in *ptr* to *ptr + nbytes*
    as hex.
    """
def is_nonelike(ty):
    """ returns if 'ty' is none """
def is_empty_tuple(ty):
    """ returns if 'ty' is an empty tuple """
def create_constant_array(ty, val):
    """
    Create an LLVM-constant of a fixed-length array from Python values.

    The type provided is the type of the elements.
    """
