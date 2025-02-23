from _typeshed import Incomplete
from numba import njit as njit
from numba.core import cgutils as cgutils, errors as errors, imputils as imputils, types as types, utils as utils
from numba.core.datamodel import default_manager as default_manager, models as models
from numba.core.registry import cpu_target as cpu_target
from numba.core.serialize import disable_pickling as disable_pickling
from numba.core.typing import templates as templates
from numba.core.typing.asnumbatype import as_numba_type as as_numba_type
from numba.experimental.jitclass import _box as _box

class InstanceModel(models.StructModel):
    def __init__(self, dmm, fe_typ) -> None: ...

class InstanceDataModel(models.StructModel):
    def __init__(self, dmm, fe_typ) -> None: ...

def _mangle_attr(name):
    """
    Mangle attributes.
    The resulting name does not startswith an underscore '_'.
    """

_ctor_template: str

def _getargs(fn_sig):
    """
    Returns list of positional and keyword argument names in order.
    """

class JitClassType(type):
    """
    The type of any jitclass.
    """
    def __new__(cls, name, bases, dct): ...
    def _set_init(cls) -> None:
        """
        Generate a wrapper for calling the constructor from pure Python.
        Note the wrapper will only accept positional arguments.
        """
    def __instancecheck__(cls, instance): ...
    def __call__(cls, *args, **kwargs): ...

def _validate_spec(spec) -> None: ...
def _fix_up_private_attr(clsname, spec):
    """
    Apply the same changes to dunder names as CPython would.
    """
def _add_linking_libs(context, call) -> None:
    """
    Add the required libs for the callable to allow inlining.
    """
def register_class_type(cls, spec, class_ctor, builder):
    """
    Internal function to create a jitclass.

    Args
    ----
    cls: the original class object (used as the prototype)
    spec: the structural specification contains the field types.
    class_ctor: the numba type to represent the jitclass
    builder: the internal jitclass builder
    """

class ConstructorTemplate(templates.AbstractTemplate):
    """
    Base class for jitclass constructor templates.
    """
    def generic(self, args, kws): ...

def _drop_ignored_attrs(dct) -> None: ...

class ClassBuilder:
    """
    A jitclass builder for a mutable jitclass.  This will register
    typing and implementation hooks to the given typing and target contexts.
    """
    class_impl_registry: Incomplete
    implemented_methods: Incomplete
    class_type: Incomplete
    typingctx: Incomplete
    targetctx: Incomplete
    def __init__(self, class_type, typingctx, targetctx) -> None: ...
    def register(self) -> None:
        """
        Register to the frontend and backend.
        """
    def _register_methods(self, registry, instance_type) -> None:
        """
        Register method implementations.
        This simply registers that the method names are valid methods.  Inside
        of imp() below we retrieve the actual method to run from the type of
        the receiver argument (i.e. self).
        """
    def _implement_method(self, registry, attr): ...

class ClassAttribute(templates.AttributeTemplate):
    key = types.ClassInstanceType
    def generic_resolve(self, instance, attr): ...

def get_attr_impl(context, builder, typ, value, attr):
    """
    Generic getattr() for @jitclass instances.
    """
def set_attr_impl(context, builder, sig, args, attr) -> None:
    """
    Generic setattr() for @jitclass instances.
    """
def imp_dtor(context, module, instance_type): ...
def ctor_impl(context, builder, sig, args):
    """
    Generic constructor (__new__) for jitclasses.
    """
