from _typeshed import Incomplete
from contextlib import contextmanager as contextmanager
from numba.core import cgutils as cgutils, types as types
from numba.core.errors import NumbaNotImplementedError as NumbaNotImplementedError, TypingError as TypingError
from numba.core.pythonapi import NativeValue as NativeValue, box as box, reflect as reflect, unbox as unbox
from numba.core.typing.typeof import Purpose as Purpose, typeof as typeof
from numba.cpython import listobj as listobj, setobj as setobj
from numba.np import numpy_support as numpy_support

def box_np_scalars(typ, val, c, py_boxing_func, py_type): ...
def box_py_bool(typ, val, c): ...
def box_np_bool(typ, val, c): ...
def unbox_boolean(typ, obj, c): ...
def box_literal_integer(typ, val, c): ...
def unbox_integer(typ, obj, c): ...
def box_py_integer(typ, val, c): ...
def box_np_integer(typ, val, c): ...
def box_py_float(typ, val, c): ...
def box_np_float(typ, val, c): ...
def unbox_float(typ, obj, c): ...
def box_py_complex(typ, val, c): ...
def unbox_py_complex(typ, obj, c): ...
def box_np_complex(typ, val, c): ...
def box_none(typ, val, c): ...
def unbox_none(typ, val, c): ...
def box_npdatetime(typ, val, c): ...
def unbox_npdatetime(typ, obj, c): ...
def box_nptimedelta(typ, val, c): ...
def unbox_nptimedelta(typ, obj, c): ...
def box_raw_pointer(typ, val, c):
    """
    Convert a raw pointer to a Python int.
    """
def box_enum(typ, val, c):
    """
    Fetch an enum member given its native value.
    """
def unbox_enum(typ, obj, c):
    """
    Convert an enum member's value to its native value.
    """
def box_undefvar(typ, val, c) -> None:
    """This type cannot be boxed, there's no Python equivalent"""
def box_record(typ, val, c): ...
def unbox_record(typ, obj, c): ...
def box_unicodecharseq(typ, val, c): ...
def unbox_unicodecharseq(typ, obj, c): ...
def box_bytes(typ, val, c): ...
def box_charseq(typ, val, c): ...
def unbox_charseq(typ, obj, c): ...
def box_optional(typ, val, c): ...
def unbox_optional(typ, obj, c):
    """
    Convert object *obj* to a native optional structure.
    """
def unbox_slice(typ, obj, c):
    """
    Convert object *obj* to a native slice structure.
    """
def box_slice_literal(typ, val, c): ...
def unbox_string_literal(typ, obj, c): ...
def box_array(typ, val, c): ...
def unbox_buffer(typ, obj, c):
    """
    Convert a Py_buffer-providing object to a native array structure.
    """
def unbox_array(typ, obj, c):
    """
    Convert a Numpy array object to a native array structure.
    """
def box_tuple(typ, val, c):
    """
    Convert native array or structure *val* to a tuple object.
    """
def box_namedtuple(typ, val, c):
    """
    Convert native array or structure *val* to a namedtuple object.
    """
def unbox_tuple(typ, obj, c):
    """
    Convert tuple *obj* to a native array (if homogeneous) or structure.
    """
def box_list(typ, val, c):
    """
    Convert native list *val* to a list object.
    """

class _NumbaTypeHelper:
    """A helper for acquiring `numba.typeof` for type checking.

    Usage
    -----

        # `c` is the boxing context.
        with _NumbaTypeHelper(c) as nth:
            # This contextmanager maintains the lifetime of the `numba.typeof`
            # function.
            the_numba_type = nth.typeof(some_object)
            # Do work on the type object
            do_checks(the_numba_type)
            # Cleanup
            c.pyapi.decref(the_numba_type)
        # At this point *nth* should not be used.
    """
    c: Incomplete
    def __init__(self, c) -> None: ...
    typeof_fn: Incomplete
    def __enter__(self): ...
    def __exit__(self, *args, **kwargs) -> None: ...
    def typeof(self, obj): ...

def _python_list_to_native(typ, obj, c, size, listptr, errorptr) -> None:
    """
    Construct a new native list from a Python list.
    """
def unbox_list(typ, obj, c):
    """
    Convert list *obj* to a native list.

    If list was previously unboxed, we reuse the existing native list
    to ensure consistency.
    """
def reflect_list(typ, val, c) -> None:
    """
    Reflect the native list's contents into the Python object.
    """
def _python_set_to_native(typ, obj, c, size, setptr, errorptr) -> None:
    """
    Construct a new native set from a Python set.
    """
def unbox_set(typ, obj, c):
    """
    Convert set *obj* to a native set.

    If set was previously unboxed, we reuse the existing native set
    to ensure consistency.
    """
def _native_set_to_python_list(typ, payload, c):
    """
    Create a Python list from a native set's items.
    """
def box_set(typ, val, c):
    """
    Convert native set *val* to a set object.
    """
def reflect_set(typ, val, c) -> None:
    """
    Reflect the native set's contents into the Python object.
    """
def box_generator(typ, val, c): ...
def unbox_generator(typ, obj, c): ...
def box_dtype(typ, val, c): ...
def unbox_dtype(typ, val, c): ...
def box_number_class(typ, val, c): ...
def unbox_number_class(typ, val, c): ...
def box_pyobject(typ, val, c): ...
def unbox_pyobject(typ, obj, c): ...
def unbox_funcptr(typ, obj, c): ...
def box_deferred(typ, val, c): ...
def unbox_deferred(typ, obj, c): ...
def unbox_dispatcher(typ, obj, c): ...
def unbox_unsupported(typ, obj, c): ...
def box_unsupported(typ, val, c): ...
def box_literal(typ, val, c): ...
def box_meminfo_pointer(typ, val, c): ...
def unbox_meminfo_pointer(typ, obj, c): ...
def unbox_typeref(typ, val, c): ...
def box_LiteralStrKeyDict(typ, val, c): ...
def unbox_numpy_random_bitgenerator(typ, obj, c):
    """
    The bit_generator instance has a `.ctypes` attr which is a namedtuple
    with the following members (types):
    * state_address (Python int)
    * state (ctypes.c_void_p)
    * next_uint64 (ctypes.CFunctionType instance)
    * next_uint32 (ctypes.CFunctionType instance)
    * next_double (ctypes.CFunctionType instance)
    * bit_generator (ctypes.c_void_p)
    """

_bit_gen_type: Incomplete

def unbox_numpy_random_generator(typ, obj, c):
    """
    Here we're creating a NumPyRandomGeneratorType StructModel with following fields:
    * ('bit_generator', _bit_gen_type): The unboxed BitGenerator associated with
                                        this Generator object instance.
    * ('parent', types.pyobject): Pointer to the original Generator PyObject.
    * ('meminfo', types.MemInfoPointer(types.voidptr)): The information about the memory
        stored at the pointer (to the original Generator PyObject). This is useful for
        keeping track of reference counts within the Python runtime. Helps prevent cases
        where deletion happens in Python runtime without NRT being awareness of it.
    """
def box_numpy_random_generator(typ, val, c): ...
