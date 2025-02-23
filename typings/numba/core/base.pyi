from _typeshed import Incomplete
from collections.abc import Generator
from functools import cached_property as cached_property
from numba import _dynfunc as _dynfunc, _helperlib as _helperlib
from numba.core import cgutils as cgutils, config as config, datamodel as datamodel, debuginfo as debuginfo, errors as errors, event as event, funcdesc as funcdesc, imputils as imputils, targetconfig as targetconfig, types as types, utils as utils
from numba.core.compiler_lock import global_compiler_lock as global_compiler_lock
from numba.core.imputils import RegistryLoader as RegistryLoader, builtin_registry as builtin_registry, impl_ret_borrowed as impl_ret_borrowed, user_function as user_function, user_generator as user_generator
from numba.core.pythonapi import PythonAPI as PythonAPI
from numba.cpython import builtins as builtins

GENERIC_POINTER: Incomplete
PYOBJECT = GENERIC_POINTER
void_ptr = GENERIC_POINTER

class OverloadSelector:
    '''
    An object matching an actual signature against a registry of formal
    signatures and choosing the best candidate, if any.

    In the current implementation:
    - a "signature" is a tuple of type classes or type instances
    - the "best candidate" is the most specific match
    '''
    versions: Incomplete
    _cache: Incomplete
    def __init__(self) -> None: ...
    def find(self, sig): ...
    def _find(self, sig): ...
    def _select_compatible(self, sig):
        """
        Select all compatible signatures and their implementation.
        """
    def _best_signature(self, candidates):
        """
        Returns the best signature out of the candidates
        """
    def _sort_signatures(self, candidates):
        """
        Sort signatures in ascending level of genericity.

        Returns a 2-tuple:

            * ordered list of signatures
            * dictionary containing genericity scores
        """
    def _match_arglist(self, formal_args, actual_args):
        '''
        Returns True if the signature is "matching".
        A formal signature is "matching" if the actual signature matches exactly
        or if the formal signature is a compatible generic signature.
        '''
    def _match(self, formal, actual): ...
    def append(self, value, sig) -> None:
        """
        Add a formal signature and its associated value.
        """

def _load_global_helpers() -> None:
    """
    Execute once to install special symbols into the LLVM symbol table.
    """

class BaseContext:
    """

    Notes on Structure
    ------------------

    Most objects are lowered as plain-old-data structure in the generated
    llvm.  They are passed around by reference (a pointer to the structure).
    Only POD structure can live across function boundaries by copying the
    data.
    """
    strict_alignment: bool
    implement_powi_as_math_call: bool
    implement_pow_as_math_call: bool
    enable_debuginfo: bool
    DIBuilder: Incomplete
    @property
    def enable_boundscheck(self): ...
    _boundscheck: Incomplete
    @enable_boundscheck.setter
    def enable_boundscheck(self, value) -> None: ...
    enable_nrt: bool
    auto_parallel: bool
    aot_mode: bool
    error_model: Incomplete
    allow_dynamic_globals: bool
    fastmath: bool
    environment: Incomplete
    fndesc: Incomplete
    address_size: Incomplete
    typing_context: Incomplete
    target_name: Incomplete
    target: Incomplete
    _registries: Incomplete
    _defns: Incomplete
    _getattrs: Incomplete
    _setattrs: Incomplete
    _casts: Incomplete
    _get_constants: Incomplete
    _generators: Incomplete
    special_ops: Incomplete
    cached_internal_func: Incomplete
    _pid: Incomplete
    _codelib_stack: Incomplete
    data_model_manager: Incomplete
    def __init__(self, typing_context, target) -> None: ...
    def init(self) -> None:
        """
        For subclasses to add initializer
        """
    def refresh(self) -> None:
        """
        Refresh context with new declarations from known registries.
        Useful for third-party extensions.
        """
    def load_additional_registries(self) -> None:
        """
        Load target-specific registries.  Can be overridden by subclasses.
        """
    def mangler(self, name, types, *, abi_tags=(), uid: Incomplete | None = None):
        """
        Perform name mangling.
        """
    def get_env_name(self, fndesc):
        """Get the environment name given a FunctionDescriptor.

        Use this instead of the ``fndesc.env_name`` so that the target-context
        can provide necessary mangling of the symbol to meet ABI requirements.
        """
    def declare_env_global(self, module, envname):
        """Declare the Environment pointer as a global of the module.

        The pointer is initialized to NULL.  It must be filled by the runtime
        with the actual address of the Env before the associated function
        can be executed.

        Parameters
        ----------
        module :
            The LLVM Module
        envname : str
            The name of the global variable.
        """
    def get_arg_packer(self, fe_args): ...
    def get_data_packer(self, fe_types): ...
    @property
    def target_data(self) -> None: ...
    @cached_property
    def nonconst_module_attrs(self):
        """
        All module attrs are constant for targets using BaseContext.
        """
    @cached_property
    def nrt(self): ...
    def subtarget(self, **kws): ...
    def install_registry(self, registry) -> None:
        """
        Install a *registry* (a imputils.Registry instance) of function
        and attribute implementations.
        """
    def insert_func_defn(self, defns) -> None: ...
    def _insert_getattr_defn(self, defns) -> None: ...
    def _insert_setattr_defn(self, defns) -> None: ...
    def _insert_cast_defn(self, defns) -> None: ...
    def _insert_get_constant_defn(self, defns) -> None: ...
    def insert_user_function(self, func, fndesc, libs=()) -> None: ...
    def insert_generator(self, genty, gendesc, libs=()) -> None: ...
    def remove_user_function(self, func) -> None:
        """
        Remove user function *func*.
        KeyError is raised if the function isn't known to us.
        """
    def get_external_function_type(self, fndesc): ...
    def declare_function(self, module, fndesc): ...
    def declare_external_function(self, module, fndesc): ...
    def insert_const_string(self, mod, string):
        """
        Insert constant *string* (a str object) into module *mod*.
        """
    def insert_const_bytes(self, mod, bytes, name: Incomplete | None = None):
        """
        Insert constant *byte* (a `bytes` object) into module *mod*.
        """
    def insert_unique_const(self, mod, name, val):
        """
        Insert a unique internal constant named *name*, with LLVM value
        *val*, into module *mod*.
        """
    def get_argument_type(self, ty): ...
    def get_return_type(self, ty): ...
    def get_data_type(self, ty):
        """
        Get a LLVM data representation of the Numba type *ty* that is safe
        for storage.  Record data are stored as byte array.

        The return value is a llvmlite.ir.Type object, or None if the type
        is an opaque pointer (???).
        """
    def get_value_type(self, ty): ...
    def pack_value(self, builder, ty, value, ptr, align: Incomplete | None = None) -> None:
        """
        Pack value into the array storage at *ptr*.
        If *align* is given, it is the guaranteed alignment for *ptr*
        (by default, the standard ABI alignment).
        """
    def unpack_value(self, builder, ty, ptr, align: Incomplete | None = None):
        """
        Unpack value from the array storage at *ptr*.
        If *align* is given, it is the guaranteed alignment for *ptr*
        (by default, the standard ABI alignment).
        """
    def get_constant_generic(self, builder, ty, val):
        """
        Return a LLVM constant representing value *val* of Numba type *ty*.
        """
    def get_constant(self, ty, val):
        """
        Same as get_constant_generic(), but without specifying *builder*.
        Works only for simple types.
        """
    def get_constant_undef(self, ty): ...
    def get_constant_null(self, ty): ...
    def get_function(self, fn, sig, _firstcall: bool = True):
        """
        Return the implementation of function *fn* for signature *sig*.
        The return value is a callable with the signature (builder, args).
        """
    def get_generator_desc(self, genty):
        """
        """
    def get_generator_impl(self, genty):
        """
        """
    def get_bound_function(self, builder, obj, ty): ...
    def get_getattr(self, typ, attr):
        """
        Get the getattr() implementation for the given type and attribute name.
        The return value is a callable with the signature
        (context, builder, typ, val, attr).
        """
    def get_setattr(self, attr, sig):
        """
        Get the setattr() implementation for the given attribute name
        and signature.
        The return value is a callable with the signature (builder, args).
        """
    def get_argument_value(self, builder, ty, val):
        """
        Argument representation to local value representation
        """
    def get_returned_value(self, builder, ty, val):
        """
        Return value representation to local value representation
        """
    def get_return_value(self, builder, ty, val):
        """
        Local value representation to return type representation
        """
    def get_value_as_argument(self, builder, ty, val):
        """Prepare local value representation as argument type representation
        """
    def get_value_as_data(self, builder, ty, val): ...
    def get_data_as_value(self, builder, ty, val): ...
    def pair_first(self, builder, val, ty):
        """
        Extract the first element of a heterogeneous pair.
        """
    def pair_second(self, builder, val, ty):
        """
        Extract the second element of a heterogeneous pair.
        """
    def cast(self, builder, val, fromty, toty):
        """
        Cast a value of type *fromty* to type *toty*.
        This implements implicit conversions as can happen due to the
        granularity of the Numba type system, or lax Python semantics.
        """
    def generic_compare(self, builder, key, argtypes, args):
        """
        Compare the given LLVM values of the given Numba types using
        the comparison *key* (e.g. '==').  The values are first cast to
        a common safe conversion type.
        """
    def make_optional_none(self, builder, valtype): ...
    def make_optional_value(self, builder, valtype, value): ...
    def is_true(self, builder, typ, val):
        """
        Return the truth value of a value of the given Numba type.
        """
    def get_c_value(self, builder, typ, name, dllimport: bool = False):
        """
        Get a global value through its C-accessible *name*, with the given
        LLVM type.
        If *dllimport* is true, the symbol will be marked as imported
        from a DLL (necessary for AOT compilation under Windows).
        """
    def call_external_function(self, builder, callee, argtys, args): ...
    def get_function_pointer_type(self, typ): ...
    def call_function_pointer(self, builder, funcptr, args, cconv: Incomplete | None = None): ...
    def print_string(self, builder, text): ...
    def debug_print(self, builder, text) -> None: ...
    def printf(self, builder, format_string, *args): ...
    def get_struct_type(self, struct):
        """
        Get the LLVM struct type for the given Structure class *struct*.
        """
    def get_dummy_value(self): ...
    def get_dummy_type(self): ...
    def _compile_subroutine_no_cache(self, builder, impl, sig, locals={}, flags: Incomplete | None = None):
        """
        Invoke the compiler to compile a function to be used inside a
        nopython function, but without generating code to call that
        function.

        Note this context's flags are not inherited.
        """
    def compile_subroutine(self, builder, impl, sig, locals={}, flags: Incomplete | None = None, caching: bool = True):
        """
        Compile the function *impl* for the given *sig* (in nopython mode).
        Return an instance of CompileResult.

        If *caching* evaluates True, the function keeps the compiled function
        for reuse in *.cached_internal_func*.
        """
    def compile_internal(self, builder, impl, sig, args, locals={}):
        """
        Like compile_subroutine(), but also call the function with the given
        *args*.
        """
    def call_internal(self, builder, fndesc, sig, args):
        """
        Given the function descriptor of an internally compiled function,
        emit a call to that function with the given arguments.
        """
    def call_internal_no_propagate(self, builder, fndesc, sig, args):
        """Similar to `.call_internal()` but does not handle or propagate
        the return status automatically.
        """
    def call_unresolved(self, builder, name, sig, args):
        '''
        Insert a function call to an unresolved symbol with the given *name*.

        Note: this is used for recursive call.

        In the mutual recursion case::

            @njit
            def foo():
                ...  # calls bar()

            @njit
            def bar():
                ... # calls foo()

            foo()

        When foo() is called, the compilation of bar() is fully completed
        (codegen\'ed and loaded) before foo() is. Since MCJIT\'s eager compilation
        doesn\'t allow loading modules with declare-only functions (which is
        needed for foo() in bar()), the call_unresolved injects a global
        variable that the "linker" can update even after the module is loaded by
        MCJIT. The linker would allocate space for the global variable before
        the bar() module is loaded. When later foo() module is defined, it will
        update bar()\'s reference to foo().

        The legacy lazy JIT and the new ORC JIT would allow a declare-only
        function be used in a module as long as it is defined by the time of its
        first use.
        '''
    def get_executable(self, func, fndesc, env) -> None: ...
    def get_python_api(self, builder): ...
    def sentry_record_alignment(self, rectyp, attr) -> None:
        """
        Assumes offset starts from a properly aligned location
        """
    def get_helper_class(self, typ, kind: str = 'value'):
        """
        Get a helper class for the given *typ*.
        """
    def _make_helper(self, builder, typ, value: Incomplete | None = None, ref: Incomplete | None = None, kind: str = 'value'): ...
    def make_helper(self, builder, typ, value: Incomplete | None = None, ref: Incomplete | None = None):
        """
        Get a helper object to access the *typ*'s members,
        for the given value or reference.
        """
    def make_data_helper(self, builder, typ, ref: Incomplete | None = None):
        """
        As make_helper(), but considers the value as stored in memory,
        rather than a live value.
        """
    def make_array(self, typ): ...
    def populate_array(self, arr, **kwargs):
        """
        Populate array structure.
        """
    def make_complex(self, builder, typ, value: Incomplete | None = None):
        """
        Get a helper object to access the given complex numbers' members.
        """
    def make_tuple(self, builder, typ, values):
        """
        Create a tuple of the given *typ* containing the *values*.
        """
    def make_constant_array(self, builder, typ, ary):
        """
        Create an array structure reifying the given constant array.
        A low-level contiguous array constant is created in the LLVM IR.
        """
    def add_dynamic_addr(self, builder, intaddr, info):
        """
        Returns dynamic address as a void pointer `i8*`.

        Internally, a global variable is added to inform the lowerer about
        the usage of dynamic addresses.  Caching will be disabled.
        """
    def get_abi_sizeof(self, ty):
        """
        Get the ABI size of LLVM type *ty*.
        """
    def get_abi_alignment(self, ty):
        """
        Get the ABI alignment of LLVM type *ty*.
        """
    def get_preferred_array_alignment(context, ty):
        """
        Get preferred array alignment for Numba type *ty*.
        """
    def post_lowering(self, mod, library) -> None:
        """Run target specific post-lowering transformation here.
        """
    def create_module(self, name) -> None:
        """Create a LLVM module

        The default implementation in BaseContext always raises a
        ``NotImplementedError`` exception. Subclasses should implement
        this method.
        """
    @property
    def active_code_library(self):
        """Get the active code library
        """
    def push_code_library(self, lib) -> Generator[None]:
        """Push the active code library for the context
        """
    def add_linking_libs(self, libs) -> None:
        """Add iterable of linking libraries to the *active_code_library*.
        """
    def get_ufunc_info(self, ufunc_key) -> None:
        """Get the ufunc implementation for a given ufunc object.

        The default implementation in BaseContext always raises a
        ``NotImplementedError`` exception. Subclasses may raise ``KeyError``
        to signal that the given ``ufunc_key`` is not available.

        Parameters
        ----------
        ufunc_key : NumPy ufunc

        Returns
        -------
        res : dict[str, callable]
            A mapping of a NumPy ufunc type signature to a lower-level
            implementation.
        """

class _wrap_impl:
    """
    A wrapper object to call an implementation function with some predefined
    (context, signature) arguments.
    The wrapper also forwards attribute queries, which is important.
    """
    _callable: Incomplete
    _imp: Incomplete
    _context: Incomplete
    _sig: Incomplete
    def __init__(self, imp, context, sig) -> None: ...
    def __call__(self, builder, args, loc: Incomplete | None = None): ...
    def __getattr__(self, item): ...
    def __repr__(self) -> str: ...

def _has_loc(fn):
    """Does function *fn* take ``loc`` argument?
    """

class _wrap_missing_loc:
    func: Incomplete
    def __init__(self, fn) -> None: ...
    def __call__(self):
        """Wrap function for missing ``loc`` keyword argument.
        Otherwise, return the original *fn*.
        """
    def __repr__(self) -> str: ...

def _initialize_llvm_lock_event() -> None:
    """Initial event triggers for LLVM lock
    """
