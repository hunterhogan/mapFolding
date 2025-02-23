from numba.core.extending import overload as overload
from numba.core.types import ClassInstanceType as ClassInstanceType

def _get_args(n_args): ...
def class_instance_overload(target):
    """
    Decorator to add an overload for target that applies when the first argument
    is a ClassInstanceType.
    """
def extract_template(template, name):
    """
    Extract a code-generated function from a string template.
    """
def register_simple_overload(func, *attrs, n_args: int = 1):
    """
    Register an overload for func that checks for methods __attr__ for each
    attr in attrs.
    """
def try_call_method(cls_type, method, n_args: int = 1):
    """
    If method is defined for cls_type, return a callable that calls this method.
    If not, return None.
    """
def try_call_complex_method(cls_type, method):
    """ __complex__ needs special treatment as the argument names are kwargs
    and therefore specific in name and default value.
    """
def take_first(*options):
    """
    Take the first non-None option.
    """
def class_bool(x): ...
def class_complex(real: int = 0, imag: int = 0): ...
def class_contains(x, y): ...
def class_float(x): ...
def class_int(x): ...
def class_str(x): ...
def class_ne(x, y): ...
def register_reflected_overload(func, meth_forward, meth_reflected): ...
