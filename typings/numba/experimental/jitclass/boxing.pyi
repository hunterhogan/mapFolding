from _typeshed import Incomplete
from numba.core import cgutils as cgutils, types as types
from numba.core.decorators import njit as njit
from numba.core.pythonapi import NativeValue as NativeValue, box as box, unbox as unbox
from numba.core.typing.typeof import typeof_impl as typeof_impl
from numba.experimental.jitclass import _box as _box

_getter_code_template: str
_setter_code_template: str
_method_code_template: str

def _generate_property(field, template, fname):
    """
    Generate simple function that get/set a field of the instance
    """

_generate_getter: Incomplete
_generate_setter: Incomplete

def _generate_method(name, func):
    """
    Generate a wrapper for calling a method.  Note the wrapper will only
    accept positional arguments.
    """

_cache_specialized_box: Incomplete

def _specialize_box(typ):
    """
    Create a subclass of Box that is specialized to the jitclass.

    This function caches the result to avoid code bloat.
    """
def _box_class_instance(typ, val, c): ...
def _unbox_class_instance(typ, val, c): ...
def _typeof_jitclass_box(val, c): ...
