from .manager import DataModelManager as DataModelManager
from _typeshed import Incomplete

def register(dmm, typecls):
    """Used as decorator to simplify datamodel registration.
    Returns the object being decorated so that chaining is possible.
    """

default_manager: Incomplete
register_default: Incomplete
