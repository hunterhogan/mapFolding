from _typeshed import Incomplete
from numba import njit as njit
from numba.core import cgutils as cgutils, imputils as imputils, types as types
from numba.core.datamodel import default_manager as default_manager, models as models
from numba.core.extending import NativeValue as NativeValue, box as box, infer_getattr as infer_getattr, intrinsic as intrinsic, lower_getattr_generic as lower_getattr_generic, lower_setattr_generic as lower_setattr_generic, overload as overload, unbox as unbox
from numba.core.typing.templates import AttributeTemplate as AttributeTemplate

class _Utils:
    """Internal builder-code utils for structref definitions.
    """
    context: Incomplete
    builder: Incomplete
    struct_type: Incomplete
    def __init__(self, context, builder, struct_type) -> None:
        """
        Parameters
        ----------
        context :
            a numba target context
        builder :
            a llvmlite IRBuilder
        struct_type : numba.core.types.StructRef
        """
    def new_struct_ref(self, mi):
        """Encapsulate the MemInfo from a `StructRefPayload` in a `StructRef`
        """
    def get_struct_ref(self, val):
        """Return a helper for accessing a StructRefType
        """
    def get_data_pointer(self, val):
        """Get the data pointer to the payload from a `StructRefType`.
        """
    def get_data_struct(self, val):
        """Get a getter/setter helper for accessing a `StructRefPayload`
        """

def define_attributes(struct_typeclass):
    """Define attributes on `struct_typeclass`.

    Defines both setters and getters in jit-code.

    This is called directly in `register()`.
    """
def define_boxing(struct_type, obj_class):
    """Define the boxing & unboxing logic for `struct_type` to `obj_class`.

    Defines both boxing and unboxing.

    - boxing turns an instance of `struct_type` into a PyObject of `obj_class`
    - unboxing turns an instance of `obj_class` into an instance of
      `struct_type` in jit-code.


    Use this directly instead of `define_proxy()` when the user does not
    want any constructor to be defined.
    """
def define_constructor(py_class, struct_typeclass, fields) -> None:
    """Define the jit-code constructor for `struct_typeclass` using the
    Python type `py_class` and the required `fields`.

    Use this instead of `define_proxy()` if the user does not want boxing
    logic defined.
    """
def define_proxy(py_class, struct_typeclass, fields) -> None:
    """Defines a PyObject proxy for a structref.

    This makes `py_class` a valid constructor for creating a instance of
    `struct_typeclass` that contains the members as defined by `fields`.

    Parameters
    ----------
    py_class : type
        The Python class for constructing an instance of `struct_typeclass`.
    struct_typeclass : numba.core.types.Type
        The structref type class to bind to.
    fields : Sequence[str]
        A sequence of field names.

    Returns
    -------
    None
    """
def register(struct_type):
    """Register a `numba.core.types.StructRef` for use in jit-code.

    This defines the data-model for lowering an instance of `struct_type`.
    This defines attributes accessor and mutator for an instance of
    `struct_type`.

    Parameters
    ----------
    struct_type : type
        A subclass of `numba.core.types.StructRef`.

    Returns
    -------
    struct_type : type
        Returns the input argument so this can act like a decorator.

    Examples
    --------

    .. code-block::

        class MyStruct(numba.core.types.StructRef):
            ...  # the simplest subclass can be empty

        numba.experimental.structref.register(MyStruct)

    """
def new(typingctx, struct_type):
    """new(struct_type)

    A jit-code only intrinsic. Used to allocate an **empty** mutable struct.
    The fields are zero-initialized and must be set manually after calling
    the function.

    Example:

        instance = new(MyStruct)
        instance.field = field_value
    """

class StructRefProxy:
    """A PyObject proxy to the Numba allocated structref data structure.

    Notes
    -----

    * Subclasses should not define ``__init__``.
    * Subclasses can override ``__new__``.
    """
    __slots__: Incomplete
    @classmethod
    def _numba_box_(cls, ty, mi):
        """Called by boxing logic, the conversion of Numba internal
        representation into a PyObject.

        Parameters
        ----------
        ty :
            a Numba type instance.
        mi :
            a wrapped MemInfoPointer.

        Returns
        -------
        instance :
             a StructRefProxy instance.
        """
    def __new__(cls, *args):
        """Construct a new instance of the structref.

        This takes positional-arguments only due to limitation of the compiler.
        The arguments are mapped to ``cls(*args)`` in jit-code.
        """
    @property
    def _numba_type_(self):
        """Returns the Numba type instance for this structref instance.

        Subclasses should NOT override.
        """
