from _typeshed import Incomplete
from numba.core import cgutils as cgutils, config as config, types as types
from numba.core.extending import intrinsic as intrinsic, make_attribute_wrapper as make_attribute_wrapper, models as models, overload as overload, register_jitable as register_jitable, register_model as register_model

class NumPyRngBitGeneratorModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None: ...

_bit_gen_type: Incomplete

class NumPyRandomGeneratorTypeModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None: ...

def _generate_next_binding(overloadable_function, return_type):
    '''
        Generate the overloads for "next_(some type)" functions.
    '''
def next_double(bitgen): ...
def next_uint32(bitgen): ...
def next_uint64(bitgen): ...
def next_float(bitgen): ...
