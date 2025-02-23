from numba.core import cgutils as cgutils, types as types
from numba.core.errors import NumbaExperimentalFeatureWarning as NumbaExperimentalFeatureWarning, NumbaValueError as NumbaValueError
from numba.extending import NativeValue as NativeValue, box as box, lower_builtin as lower_builtin, make_attribute_wrapper as make_attribute_wrapper, models as models, register_model as register_model, type_callable as type_callable, unbox as unbox

class PolynomialModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None: ...

def type_polynomial(context): ...
def impl_polynomial1(context, builder, sig, args): ...
def impl_polynomial3(context, builder, sig, args): ...
def unbox_polynomial(typ, obj, c):
    """
    Convert a Polynomial object to a native polynomial structure.
    """
def box_polynomial(typ, val, c):
    """
    Convert a native polynomial structure to a Polynomial object.
    """
