from _typeshed import Incomplete
from numba.core import types as types

class DataModelManager:
    """Manages mapping of FE types to their corresponding data model
    """
    _handlers: Incomplete
    _cache: Incomplete
    def __init__(self, handlers: Incomplete | None = None) -> None:
        """
        Parameters
        -----------
        handlers: Mapping[Type, DataModel] or None
            Optionally provide the initial handlers mapping.
        """
    def register(self, fetypecls, handler) -> None:
        """Register the datamodel factory corresponding to a frontend-type class
        """
    def lookup(self, fetype):
        """Returns the corresponding datamodel given the frontend-type instance
        """
    def __getitem__(self, fetype):
        """Shorthand for lookup()
        """
    def copy(self):
        """
        Make a copy of the manager.
        Use this to inherit from the default data model and specialize it
        for custom target.
        """
    def chain(self, other_manager):
        """Create a new DataModelManager by chaining the handlers mapping of
        `other_manager` with a fresh handlers mapping.

        Any existing and new handlers inserted to `other_manager` will be
        visible to the new manager. Any handlers inserted to the new manager
        can override existing handlers in `other_manager` without actually
        mutating `other_manager`.

        Parameters
        ----------
        other_manager: DataModelManager
        """
