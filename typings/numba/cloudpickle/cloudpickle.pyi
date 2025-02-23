import pickle
from _typeshed import Incomplete
from collections.abc import Generator

DEFAULT_PROTOCOL: Incomplete
_PICKLE_BY_VALUE_MODULES: Incomplete
_DYNAMIC_CLASS_TRACKER_BY_CLASS: Incomplete
_DYNAMIC_CLASS_TRACKER_BY_ID: Incomplete
_DYNAMIC_CLASS_TRACKER_LOCK: Incomplete
_DYNAMIC_CLASS_TRACKER_REUSING: Incomplete
PYPY: Incomplete
builtin_code_type: Incomplete
_extract_code_globals_cache: Incomplete

def _get_or_create_tracker_id(class_def): ...
def _lookup_class_or_track(class_tracker_id, class_def): ...
def register_pickle_by_value(module) -> None:
    """Register a module to make it functions and classes picklable by value.

    By default, functions and classes that are attributes of an importable
    module are to be pickled by reference, that is relying on re-importing
    the attribute from the module at load time.

    If `register_pickle_by_value(module)` is called, all its functions and
    classes are subsequently to be pickled by value, meaning that they can
    be loaded in Python processes where the module is not importable.

    This is especially useful when developing a module in a distributed
    execution environment: restarting the client Python process with the new
    source code is enough: there is no need to re-install the new version
    of the module on all the worker nodes nor to restart the workers.

    Note: this feature is considered experimental. See the cloudpickle
    README.md file for more details and limitations.
    """
def unregister_pickle_by_value(module) -> None:
    """Unregister that the input module should be pickled by value."""
def list_registry_pickle_by_value(): ...
def _is_registered_pickle_by_value(module): ...
def _whichmodule(obj, name):
    """Find the module an object belongs to.

    This function differs from ``pickle.whichmodule`` in two ways:
    - it does not mangle the cases where obj's module is __main__ and obj was
      not found in any module.
    - Errors arising during module introspection are ignored, as those errors
      are considered unwanted side effects.
    """
def _should_pickle_by_reference(obj, name: Incomplete | None = None):
    """Test whether an function or a class should be pickled by reference

    Pickling by reference means by that the object (typically a function or a
    class) is an attribute of a module that is assumed to be importable in the
    target Python environment. Loading will therefore rely on importing the
    module and then calling `getattr` on it to access the function or class.

    Pickling by reference is the only option to pickle functions and classes
    in the standard library. In cloudpickle the alternative option is to
    pickle by value (for instance for interactively or locally defined
    functions and classes or for attributes of modules that have been
    explicitly registered to be pickled by value.
    """
def _lookup_module_and_qualname(obj, name: Incomplete | None = None): ...
def _extract_code_globals(co):
    """Find all globals names read or written to by codeblock co."""
def _find_imported_submodules(code, top_level_dependencies):
    """Find currently imported submodules used by a function.

    Submodules used by a function need to be detected and referenced for the
    function to work correctly at depickling time. Because submodules can be
    referenced as attribute of their parent package (``package.submodule``), we
    need a special introspection technique that does not rely on GLOBAL-related
    opcodes to find references of them in a code object.

    Example:
    ```
    import concurrent.futures
    import cloudpickle
    def func():
        x = concurrent.futures.ThreadPoolExecutor
    if __name__ == '__main__':
        cloudpickle.dumps(func)
    ```
    The globals extracted by cloudpickle in the function's state include the
    concurrent package, but not its submodule (here, concurrent.futures), which
    is the module used by func. Find_imported_submodules will detect the usage
    of concurrent.futures. Saving this module alongside with func will ensure
    that calling func once depickled does not fail due to concurrent.futures
    not being imported
    """

STORE_GLOBAL: Incomplete
DELETE_GLOBAL: Incomplete
LOAD_GLOBAL: Incomplete
GLOBAL_OPS: Incomplete
HAVE_ARGUMENT: Incomplete
EXTENDED_ARG: Incomplete
_BUILTIN_TYPE_NAMES: Incomplete

def _builtin_type(name): ...
def _walk_global_ops(code) -> Generator[Incomplete]:
    """Yield referenced name for global-referencing instructions in code."""
def _extract_class_dict(cls):
    """Retrieve a copy of the dict of a class without the inherited method."""
def is_tornado_coroutine(func):
    """Return whether `func` is a Tornado coroutine function.

    Running coroutines are not supported.
    """
def subimport(name): ...
def dynamic_subimport(name, vars): ...
def _get_cell_contents(cell): ...
def instance(cls):
    """Create a new instance of a class.

    Parameters
    ----------
    cls : type
        The class to create an instance of.

    Returns
    -------
    instance : cls
        A new instance of ``cls``.
    """

class _empty_cell_value:
    """Sentinel for empty closures."""
    @classmethod
    def __reduce__(cls): ...

def _make_function(code, globals, name, argdefs, closure): ...
def _make_empty_cell(): ...
def _make_cell(value=...): ...
def _make_skeleton_class(type_constructor, name, bases, type_kwargs, class_tracker_id, extra):
    '''Build dynamic class with an empty __dict__ to be filled once memoized

    If class_tracker_id is not None, try to lookup an existing class definition
    matching that id. If none is found, track a newly reconstructed class
    definition under that id so that other instances stemming from the same
    class id will also reuse this class definition.

    The "extra" variable is meant to be a dict (or None) that can be used for
    forward compatibility shall the need arise.
    '''
def _make_skeleton_enum(bases, name, qualname, members, module, class_tracker_id, extra):
    '''Build dynamic enum with an empty __dict__ to be filled once memoized

    The creation of the enum class is inspired by the code of
    EnumMeta._create_.

    If class_tracker_id is not None, try to lookup an existing enum definition
    matching that id. If none is found, track a newly reconstructed enum
    definition under that id so that other instances stemming from the same
    class id will also reuse this enum definition.

    The "extra" variable is meant to be a dict (or None) that can be used for
    forward compatibility shall the need arise.
    '''
def _make_typevar(name, bound, constraints, covariant, contravariant, class_tracker_id): ...
def _decompose_typevar(obj): ...
def _typevar_reduce(obj): ...
def _get_bases(typ): ...
def _make_dict_keys(obj, is_ordered: bool = False): ...
def _make_dict_values(obj, is_ordered: bool = False): ...
def _make_dict_items(obj, is_ordered: bool = False): ...
def _class_getnewargs(obj): ...
def _enum_getnewargs(obj): ...
def _file_reconstructor(retval): ...
def _function_getstate(func): ...
def _class_getstate(obj): ...
def _enum_getstate(obj): ...
def _code_reduce(obj):
    """code object reducer."""
def _cell_reduce(obj):
    """Cell (containing values of a function's free variables) reducer."""
def _classmethod_reduce(obj): ...
def _file_reduce(obj):
    """Save a file."""
def _getset_descriptor_reduce(obj): ...
def _mappingproxy_reduce(obj): ...
def _memoryview_reduce(obj): ...
def _module_reduce(obj): ...
def _method_reduce(obj): ...
def _logger_reduce(obj): ...
def _root_logger_reduce(obj): ...
def _property_reduce(obj): ...
def _weakset_reduce(obj): ...
def _dynamic_class_reduce(obj):
    """Save a class that can't be referenced as a module attribute.

    This method is used to serialize classes that are defined inside
    functions, or that otherwise can't be serialized as attribute lookups
    from importable modules.
    """
def _class_reduce(obj):
    """Select the reducer depending on the dynamic nature of the class obj."""
def _dict_keys_reduce(obj): ...
def _dict_values_reduce(obj): ...
def _dict_items_reduce(obj): ...
def _odict_keys_reduce(obj): ...
def _odict_values_reduce(obj): ...
def _odict_items_reduce(obj): ...
def _dataclass_field_base_reduce(obj): ...
def _function_setstate(obj, state) -> None:
    """Update the state of a dynamic function.

    As __closure__ and __globals__ are readonly attributes of a function, we
    cannot rely on the native setstate routine of pickle.load_build, that calls
    setattr on items of the slotstate. Instead, we have to modify them inplace.
    """
def _class_setstate(obj, state): ...

_DATACLASSE_FIELD_TYPE_SENTINELS: Incomplete

def _get_dataclass_field_type_sentinel(name): ...

class Pickler(pickle.Pickler):
    _dispatch_table: Incomplete
    dispatch_table: Incomplete
    def _dynamic_function_reduce(self, func):
        """Reduce a function that is not pickleable via attribute lookup."""
    def _function_reduce(self, obj):
        """Reducer for function objects.

        If obj is a top-level attribute of a file-backed module, this reducer
        returns NotImplemented, making the cloudpickle.Pickler fall back to
        traditional pickle.Pickler routines to save obj. Otherwise, it reduces
        obj using a custom cloudpickle reducer designed specifically to handle
        dynamic functions.
        """
    def _function_getnewargs(self, func): ...
    def dump(self, obj): ...
    globals_ref: Incomplete
    proto: Incomplete
    def __init__(self, file, protocol: Incomplete | None = None, buffer_callback: Incomplete | None = None) -> None: ...
    dispatch = dispatch_table
    def reducer_override(self, obj):
        """Type-agnostic reducing callback for function and classes.

            For performance reasons, subclasses of the C `pickle.Pickler` class
            cannot register custom reducers for functions and classes in the
            dispatch_table attribute. Reducers for such types must instead
            implemented via the special `reducer_override` method.

            Note that this method will be called for any object except a few
            builtin-types (int, lists, dicts etc.), which differs from reducers
            in the Pickler's dispatch_table, each of them being invoked for
            objects of a specific type only.

            This property comes in handy for classes: although most classes are
            instances of the ``type`` metaclass, some of them can be instances
            of other custom metaclasses (such as enum.EnumMeta for example). In
            particular, the metaclass will likely not be known in advance, and
            thus cannot be special-cased using an entry in the dispatch_table.
            reducer_override, among other things, allows us to register a
            reducer that will be called for any class, independently of its
            type.

            Notes:

            * reducer_override has the priority over dispatch_table-registered
            reducers.
            * reducer_override can be used to fix other limitations of
              cloudpickle for other types that suffered from type-specific
              reducers, such as Exceptions. See
              https://github.com/cloudpipe/cloudpickle/issues/248
            """
    def _save_reduce_pickle5(self, func, args, state: Incomplete | None = None, listitems: Incomplete | None = None, dictitems: Incomplete | None = None, state_setter: Incomplete | None = None, obj: Incomplete | None = None) -> None: ...
    def save_global(self, obj, name: Incomplete | None = None, pack=...):
        """Main dispatch method.

            The name of this method is somewhat misleading: all types get
            dispatched here.
            """
    def save_function(self, obj, name: Incomplete | None = None):
        """Registered with the dispatch to handle all function types.

            Determines what kind of function obj is (e.g. lambda, defined at
            interactive prompt, etc) and handles the pickling appropriately.
            """
    def save_pypy_builtin_func(self, obj) -> None:
        """Save pypy equivalent of builtin functions.

            PyPy does not have the concept of builtin-functions. Instead,
            builtin-functions are simple function instances, but with a
            builtin-code attribute.
            Most of the time, builtin functions should be pickled by attribute.
            But PyPy has flaky support for __qualname__, so some builtin
            functions such as float.__new__ will be classified as dynamic. For
            this reason only, we created this special routine. Because
            builtin-functions are not expected to have closure or globals,
            there is no additional hack (compared the one already implemented
            in pickle) to protect ourselves from reference cycles. A simple
            (reconstructor, newargs, obj.__dict__) tuple is save_reduced.  Note
            also that PyPy improved their support for __qualname__ in v3.6, so
            this routing should be removed when cloudpickle supports only PyPy
            3.6 and later.
            """

def dump(obj, file, protocol: Incomplete | None = None, buffer_callback: Incomplete | None = None) -> None:
    """Serialize obj as bytes streamed into file

    protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
    pickle.HIGHEST_PROTOCOL. This setting favors maximum communication
    speed between processes running the same Python version.

    Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
    compatibility with older versions of Python (although this is not always
    guaranteed to work because cloudpickle relies on some internal
    implementation details that can change from one Python version to the
    next).
    """
def dumps(obj, protocol: Incomplete | None = None, buffer_callback: Incomplete | None = None):
    """Serialize obj as a string of bytes allocated in memory

    protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
    pickle.HIGHEST_PROTOCOL. This setting favors maximum communication
    speed between processes running the same Python version.

    Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
    compatibility with older versions of Python (although this is not always
    guaranteed to work because cloudpickle relies on some internal
    implementation details that can change from one Python version to the
    next).
    """

load: Incomplete
loads: Incomplete
CloudPickler = Pickler
