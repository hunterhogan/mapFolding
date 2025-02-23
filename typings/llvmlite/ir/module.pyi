from _typeshed import Incomplete
from llvmlite.ir import _utils as _utils, context as context, types as types, values as values

class Module:
    context: Incomplete
    name: Incomplete
    data_layout: str
    scope: Incomplete
    triple: str
    globals: Incomplete
    metadata: Incomplete
    namedmetadata: Incomplete
    _metadatacache: Incomplete
    def __init__(self, name: str = '', context=...) -> None: ...
    def _fix_metadata_operands(self, operands): ...
    def _fix_di_operands(self, operands): ...
    def add_metadata(self, operands):
        """
        Add an unnamed metadata to the module with the given *operands*
        (a sequence of values) or return a previous equivalent metadata.
        A MDValue instance is returned, it can then be associated to
        e.g. an instruction.
        """
    def add_debug_info(self, kind, operands, is_distinct: bool = False):
        '''
        Add debug information metadata to the module with the given
        *operands* (a dict of values with string keys) or return
        a previous equivalent metadata.  *kind* is a string of the
        debug information kind (e.g. "DICompileUnit").

        A DIValue instance is returned, it can then be associated to e.g.
        an instruction.
        '''
    def add_named_metadata(self, name, element: Incomplete | None = None):
        '''
        Add a named metadata node to the module, if it doesn\'t exist,
        or return the existing node.
        If *element* is given, it will append a new element to
        the named metadata node.  If *element* is a sequence of values
        (rather than a metadata value), a new unnamed node will first be
        created.

        Example::
            module.add_named_metadata("llvm.ident", ["llvmlite/1.0"])
        '''
    def get_named_metadata(self, name):
        """
        Return the metadata node with the given *name*.  KeyError is raised
        if no such node exists (contrast with add_named_metadata()).
        """
    @property
    def functions(self):
        """
        A list of functions declared or defined in this module.
        """
    @property
    def global_values(self):
        """
        An iterable of global values in this module.
        """
    def get_global(self, name):
        """
        Get a global value by name.
        """
    def add_global(self, globalvalue) -> None:
        """
        Add a new global value.
        """
    def get_unique_name(self, name: str = ''):
        """
        Get a unique global name with the following *name* hint.
        """
    def declare_intrinsic(self, intrinsic, tys=(), fnty: Incomplete | None = None): ...
    def get_identified_types(self): ...
    def _get_body_lines(self): ...
    def _get_metadata_lines(self): ...
    def _stringify_body(self): ...
    def _stringify_metadata(self): ...
    def __repr__(self) -> str: ...
