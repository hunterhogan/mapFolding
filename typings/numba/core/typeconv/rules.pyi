from .typeconv import TypeCastingRules as TypeCastingRules, TypeManager as TypeManager
from _typeshed import Incomplete
from numba.core import config as config, types as types

default_type_manager: Incomplete

def dump_number_rules() -> None: ...
def _init_casting_rules(tm): ...

default_casting_rules: Incomplete
