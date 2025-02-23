import abc
from _typeshed import Incomplete
from abc import abstractmethod
from numba import _dispatcher as _dispatcher
from numba.core import compiler as compiler, config as config, entrypoints as entrypoints, errors as errors, serialize as serialize, sigutils as sigutils, types as types, typing as typing, utils as utils
from numba.core.bytecode import get_code_object as get_code_object
from numba.core.caching import FunctionCache as FunctionCache, NullCache as NullCache
from numba.core.compiler_lock import global_compiler_lock as global_compiler_lock
from numba.core.typeconv.rules import default_type_manager as default_type_manager
from numba.core.typing.templates import fold_arguments as fold_arguments
from numba.core.typing.typeof import Purpose as Purpose, typeof as typeof
from typing import NamedTuple

class OmittedArg:
    """
    A placeholder for omitted arguments with a default value.
    """
    value: Incomplete
    def __init__(self, value) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def _numba_type_(self): ...

class _FunctionCompiler:
    py_func: Incomplete
    targetdescr: Incomplete
    targetoptions: Incomplete
    locals: Incomplete
    pysig: Incomplete
    pipeline_class: Incomplete
    _failed_cache: Incomplete
    def __init__(self, py_func, targetdescr, targetoptions, locals, pipeline_class) -> None: ...
    def fold_argument_types(self, args, kws):
        """
        Given positional and named argument types, fold keyword arguments
        and resolve defaults by inserting types.Omitted() instances.

        A (pysig, argument types) tuple is returned.
        """
    def compile(self, args, return_type): ...
    def _compile_cached(self, args, return_type): ...
    def _compile_core(self, args, return_type): ...
    def get_globals_for_reduction(self): ...
    def _get_implementation(self, args, kws): ...
    def _customize_flags(self, flags): ...

class _GeneratedFunctionCompiler(_FunctionCompiler):
    impls: Incomplete
    def __init__(self, py_func, targetdescr, targetoptions, locals, pipeline_class) -> None: ...
    def get_globals_for_reduction(self): ...
    def _get_implementation(self, args, kws): ...

class _CompileStats(NamedTuple):
    cache_path: Incomplete
    cache_hits: Incomplete
    cache_misses: Incomplete

class CompilingCounter:
    """
    A simple counter that increment in __enter__ and decrement in __exit__.
    """
    counter: int
    def __init__(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args, **kwargs) -> None: ...
    def __bool__(self) -> bool: ...
    __nonzero__ = __bool__

class _DispatcherBase(_dispatcher.Dispatcher):
    """
    Common base class for dispatcher Implementations.
    """
    __numba__: str
    _tm: Incomplete
    overloads: Incomplete
    py_func: Incomplete
    func_code: Incomplete
    __code__: Incomplete
    _types_active_call: Incomplete
    __defaults__: Incomplete
    doc: Incomplete
    _compiling_counter: Incomplete
    _enable_sysmon: Incomplete
    def __init__(self, arg_count, py_func, pysig, can_fallback, exact_match_required) -> None: ...
    def _compilation_chain_init_hook(self) -> None:
        """
        This will be called ahead of any part of compilation taking place (this
        even includes being ahead of working out the types of the arguments).
        This permits activities such as initialising extension entry points so
        that the compiler knows about additional externally defined types etc
        before it does anything.
        """
    def _reset_overloads(self) -> None: ...
    def _make_finalizer(self):
        """
        Return a finalizer function that will release references to
        related compiled functions.
        """
    @property
    def signatures(self):
        """
        Returns a list of compiled function signatures.
        """
    @property
    def nopython_signatures(self): ...
    _can_compile: Incomplete
    def disable_compile(self, val: bool = True) -> None:
        """Disable the compilation of new signatures at call time.
        """
    def add_overload(self, cres) -> None: ...
    def fold_argument_types(self, args, kws): ...
    def get_call_template(self, args, kws):
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This allows to resolve the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
    def get_overload(self, sig):
        """
        Return the compiled function for the given signature.
        """
    @property
    def is_compiling(self):
        """
        Whether a specialization is currently being compiled.
        """
    def _compile_for_args(self, *args, **kws):
        """
        For internal use.  Compile a specialized version of the function
        for the given *args* and *kws*, and return the resulting callable.
        """
    def inspect_llvm(self, signature: Incomplete | None = None):
        """Get the LLVM intermediate representation generated by compilation.

        Parameters
        ----------
        signature : tuple of numba types, optional
            Specify a signature for which to obtain the LLVM IR. If None, the
            IR is returned for all available signatures.

        Returns
        -------
        llvm : dict[signature, str] or str
            Either the LLVM IR string for the specified signature, or, if no
            signature was given, a dictionary mapping signatures to LLVM IR
            strings.
        """
    def inspect_asm(self, signature: Incomplete | None = None):
        """Get the generated assembly code.

        Parameters
        ----------
        signature : tuple of numba types, optional
            Specify a signature for which to obtain the assembly code. If
            None, the assembly code is returned for all available signatures.

        Returns
        -------
        asm : dict[signature, str] or str
            Either the assembly code for the specified signature, or, if no
            signature was given, a dictionary mapping signatures to assembly
            code.
        """
    def inspect_types(self, file: Incomplete | None = None, signature: Incomplete | None = None, pretty: bool = False, style: str = 'default', **kwargs):
        """Print/return Numba intermediate representation (IR)-annotated code.

        Parameters
        ----------
        file : file-like object, optional
            File to which to print. Defaults to sys.stdout if None. Must be
            None if ``pretty=True``.
        signature : tuple of numba types, optional
            Print/return the intermediate representation for only the given
            signature. If None, the IR is printed for all available signatures.
        pretty : bool, optional
            If True, an Annotate object will be returned that can render the
            IR with color highlighting in Jupyter and IPython. ``file`` must
            be None if ``pretty`` is True. Additionally, the ``pygments``
            library must be installed for ``pretty=True``.
        style : str, optional
            Choose a style for rendering. Ignored if ``pretty`` is ``False``.
            This is directly consumed by ``pygments`` formatters. To see a
            list of available styles, import ``pygments`` and run
            ``list(pygments.styles.get_all_styles())``.

        Returns
        -------
        annotated : Annotate object, optional
            Only returned if ``pretty=True``, otherwise this function is only
            used for its printing side effect. If ``pretty=True``, an Annotate
            object is returned that can render itself in Jupyter and IPython.
        """
    def inspect_cfg(self, signature: Incomplete | None = None, show_wrapper: Incomplete | None = None, **kwargs):
        '''
        For inspecting the CFG of the function.

        By default the CFG of the user function is shown.  The *show_wrapper*
        option can be set to "python" or "cfunc" to show the python wrapper
        function or the *cfunc* wrapper function, respectively.

        Parameters accepted in kwargs
        -----------------------------
        filename : string, optional
            the name of the output file, if given this will write the output to
            filename
        view : bool, optional
            whether to immediately view the optional output file
        highlight : bool, set, dict, optional
            what, if anything, to highlight, options are:
            { incref : bool, # highlight NRT_incref calls
              decref : bool, # highlight NRT_decref calls
              returns : bool, # highlight exits which are normal returns
              raises : bool, # highlight exits which are from raise
              meminfo : bool, # highlight calls to NRT*meminfo
              branches : bool, # highlight true/false branches
             }
            Default is True which sets all of the above to True. Supplying a set
            of strings is also accepted, these are interpreted as key:True with
            respect to the above dictionary. e.g. {\'incref\', \'decref\'} would
            switch on highlighting on increfs and decrefs.
        interleave: bool, set, dict, optional
            what, if anything, to interleave in the LLVM IR, options are:
            { python: bool # interleave python source code with the LLVM IR
              lineinfo: bool # interleave line information markers with the LLVM
                             # IR
            }
            Default is True which sets all of the above to True. Supplying a set
            of strings is also accepted, these are interpreted as key:True with
            respect to the above dictionary. e.g. {\'python\',} would
            switch on interleaving of python source code in the LLVM IR.
        strip_ir : bool, optional
            Default is False. If set to True all LLVM IR that is superfluous to
            that requested in kwarg `highlight` will be removed.
        show_key : bool, optional
            Default is True. Create a "key" for the highlighting in the rendered
            CFG.
        fontsize : int, optional
            Default is 8. Set the fontsize in the output to this value.
        '''
    def inspect_disasm_cfg(self, signature: Incomplete | None = None):
        """
        For inspecting the CFG of the disassembly of the function.

        Requires python package: r2pipe
        Requires radare2 binary on $PATH.
        Notebook rendering requires python package: graphviz

        signature : tuple of Numba types, optional
            Print/return the disassembly CFG for only the given signatures.
            If None, the IR is printed for all available signatures.
        """
    def get_annotation_info(self, signature: Incomplete | None = None):
        """
        Gets the annotation information for the function specified by
        signature. If no signature is supplied a dictionary of signature to
        annotation information is returned.
        """
    def _explain_ambiguous(self, *args, **kws) -> None:
        """
        Callback for the C _Dispatcher object.
        """
    def _explain_matching_error(self, *args, **kws) -> None:
        """
        Callback for the C _Dispatcher object.
        """
    def _search_new_conversions(self, *args, **kws):
        """
        Callback for the C _Dispatcher object.
        Search for approximately matching signatures for the given arguments,
        and ensure the corresponding conversions are registered in the C++
        type manager.
        """
    def __repr__(self) -> str: ...
    def typeof_pyval(self, val):
        """
        Resolve the Numba type of Python value *val*.
        This is called from numba._dispatcher as a fallback if the native code
        cannot decide the type.
        """
    def _callback_add_timer(self, duration, cres, lock_name) -> None: ...
    def _callback_add_compiler_timer(self, duration, cres): ...
    def _callback_add_llvm_timer(self, duration, cres): ...

class _MemoMixin:
    __uuid: Incomplete
    _memo: Incomplete
    _recent: Incomplete
    @property
    def _uuid(self):
        """
        An instance-specific UUID, to avoid multiple deserializations of
        a given instance.

        Note: this is lazily-generated, for performance reasons.
        """
    def _set_uuid(self, u) -> None: ...

class Dispatcher(serialize.ReduceMixin, _MemoMixin, _DispatcherBase):
    """
    Implementation of user-facing dispatcher objects (i.e. created using
    the @jit decorator).
    This is an abstract base class. Subclasses should define the targetdescr
    class attribute.
    """
    _fold_args: bool
    __numba__: str
    typingctx: Incomplete
    targetctx: Incomplete
    targetoptions: Incomplete
    locals: Incomplete
    _cache: Incomplete
    _compiler: Incomplete
    _cache_hits: Incomplete
    _cache_misses: Incomplete
    _type: Incomplete
    def __init__(self, py_func, locals={}, targetoptions={}, pipeline_class=...) -> None:
        """
        Parameters
        ----------
        py_func: function object to be compiled
        locals: dict, optional
            Mapping of local variable names to Numba types.  Used to override
            the types deduced by the type inference engine.
        targetoptions: dict, optional
            Target-specific config options.
        pipeline_class: type numba.compiler.CompilerBase
            The compiler pipeline type.
        """
    def dump(self, tab: str = '') -> None: ...
    @property
    def _numba_type_(self): ...
    def enable_caching(self) -> None: ...
    def __get__(self, obj, objtype: Incomplete | None = None):
        """Allow a JIT function to be bound as a method to an object"""
    def _reduce_states(self):
        """
        Reduce the instance for pickling.  This will serialize
        the original function as well the compilation options and
        compiled signatures, but not the compiled code itself.

        NOTE: part of ReduceMixin protocol
        """
    _can_compile: Incomplete
    @classmethod
    def _rebuild(cls, uuid, py_func, locals, targetoptions, can_compile, sigs):
        """
        Rebuild an Dispatcher instance after it was __reduce__'d.

        NOTE: part of ReduceMixin protocol
        """
    def compile(self, sig): ...
    def get_compile_result(self, sig):
        """Compile (if needed) and return the compilation result with the
        given signature.

        Returns ``CompileResult``.
        Raises ``NumbaError`` if the signature is incompatible.
        """
    def recompile(self) -> None:
        """
        Recompile all signatures afresh.
        """
    @property
    def stats(self): ...
    def parallel_diagnostics(self, signature: Incomplete | None = None, level: int = 1) -> None:
        """
        Print parallel diagnostic information for the given signature. If no
        signature is present it is printed for all known signatures. level is
        used to adjust the verbosity, level=1 (default) is minimal verbosity,
        and 2, 3, and 4 provide increasing levels of verbosity.
        """
    def get_metadata(self, signature: Incomplete | None = None):
        """
        Obtain the compilation metadata for a given signature.
        """
    def get_function_type(self):
        """Return unique function type of dispatcher when possible, otherwise
        return None.

        A Dispatcher instance has unique function type when it
        contains exactly one compilation result and its compilation
        has been disabled (via its disable_compile method).
        """

class LiftedCode(serialize.ReduceMixin, _MemoMixin, _DispatcherBase, metaclass=abc.ABCMeta):
    """
    Implementation of the hidden dispatcher objects used for lifted code
    (a lifted loop is really compiled as a separate function).
    """
    _fold_args: bool
    can_cache: bool
    func_ir: Incomplete
    lifted_from: Incomplete
    typingctx: Incomplete
    targetctx: Incomplete
    flags: Incomplete
    locals: Incomplete
    def __init__(self, func_ir, typingctx, targetctx, flags, locals) -> None: ...
    def _reduce_states(self):
        """
        Reduce the instance for pickling.  This will serialize
        the original function as well the compilation options and
        compiled signatures, but not the compiled code itself.

        NOTE: part of ReduceMixin protocol
        """
    def _reduce_extras(self):
        """
        NOTE: sub-class can override to add extra states
        """
    @classmethod
    def _rebuild(cls, uuid, func_ir, flags, locals, extras):
        """
        Rebuild an Dispatcher instance after it was __reduce__'d.

        NOTE: part of ReduceMixin protocol
        """
    def get_source_location(self):
        """Return the starting line number of the loop.
        """
    def _pre_compile(self, args, return_type, flags) -> None:
        """Pre-compile actions
        """
    @abstractmethod
    def compile(self, sig):
        """Lifted code should implement a compilation method that will return
        a CompileResult.entry_point for the given signature."""
    def _get_dispatcher_for_current_target(self): ...

class LiftedLoop(LiftedCode):
    def _pre_compile(self, args, return_type, flags) -> None: ...
    def compile(self, sig): ...

class LiftedWith(LiftedCode):
    can_cache: bool
    def _reduce_extras(self): ...
    @property
    def _numba_type_(self): ...
    def get_call_template(self, args, kws):
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This enables the resolving of the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
    def compile(self, sig): ...

class ObjModeLiftedWith(LiftedWith):
    output_types: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def _numba_type_(self): ...
    def get_call_template(self, args, kws):
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This enables the resolving of the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
    def _legalize_arg_types(self, args) -> None: ...
    def compile(self, sig): ...
