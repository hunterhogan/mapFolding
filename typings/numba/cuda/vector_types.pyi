from _typeshed import Incomplete
from numba import types as types
from numba.core import cgutils as cgutils
from numba.core.extending import make_attribute_wrapper as make_attribute_wrapper, models as models, register_model as register_model
from numba.core.typing.templates import ConcreteTemplate as ConcreteTemplate, signature as signature
from numba.cuda import stubs as stubs
from numba.cuda.errors import CudaLoweringError as CudaLoweringError

typing_registry: Incomplete
impl_registry: Incomplete
register: Incomplete
register_attr: Incomplete
register_global: Incomplete
lower: Incomplete

class VectorType(types.Type):
    _base_type: Incomplete
    _attr_names: Incomplete
    _user_facing_object: Incomplete
    def __init__(self, name, base_type, attr_names, user_facing_object) -> None: ...
    @property
    def base_type(self): ...
    @property
    def attr_names(self): ...
    @property
    def num_elements(self): ...
    @property
    def user_facing_object(self): ...

def make_vector_type(name: str, base_type: types.Type, attr_names: tuple[str, ...], user_facing_object) -> types.Type:
    """Create a vector type.

    Parameters
    ----------
    name: str
        The name of the type.
    base_type: numba.types.Type
        The primitive type for each element in the vector.
    attr_names: tuple of str
        Name for each attribute.
    user_facing_object: object
        The handle to be used in cuda kernel.
    """
def enable_vector_type_ctor(vector_type: VectorType, overloads: list[list[types.Type]]):
    """Create typing and lowering for vector type constructor.

    Parameters
    ----------
    vector_type: VectorType
        The type whose constructor to type and lower.
    overloads: List of argument types
        A list containing different overloads of the constructor. Each base type
        in the argument list should either be primitive type or VectorType.
    """

vector_types: dict[str, VectorType]

def build_constructor_overloads(base_type, vty_name, num_elements, arglists, l) -> None:
    """
    For a given vector type, build a list of overloads for its constructor.
    """
def _initialize() -> None:
    """
    Construct the vector types, populate `vector_types` dictionary, and
    enable the constructors.
    """
