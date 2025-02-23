from _typeshed import Incomplete
from numba.core import bytecode as bytecode, callconv as callconv, config as config, cpu as cpu, errors as errors, interpreter as interpreter, postproc as postproc
from numba.core.compiler_machinery import PassManager as PassManager
from numba.core.environment import lookup_environment as lookup_environment
from numba.core.errors import CompilerError as CompilerError
from numba.core.object_mode_passes import ObjectModeBackEnd as ObjectModeBackEnd, ObjectModeFrontEnd as ObjectModeFrontEnd
from numba.core.targetconfig import ConfigStack as ConfigStack, Option as Option, TargetConfig as TargetConfig
from numba.core.tracing import event as event
from numba.core.typed_passes import AnnotateTypes as AnnotateTypes, DumpParforDiagnostics as DumpParforDiagnostics, IRLegalization as IRLegalization, InlineOverloads as InlineOverloads, NativeLowering as NativeLowering, NativeParforLowering as NativeParforLowering, NoPythonBackend as NoPythonBackend, NoPythonSupportedFeatureValidation as NoPythonSupportedFeatureValidation, NopythonRewrites as NopythonRewrites, NopythonTypeInference as NopythonTypeInference, ParforFusionPass as ParforFusionPass, ParforPass as ParforPass, ParforPreLoweringPass as ParforPreLoweringPass, PreLowerStripPhis as PreLowerStripPhis, PreParforPass as PreParforPass
from numba.core.untyped_passes import CanonicalizeLoopEntry as CanonicalizeLoopEntry, CanonicalizeLoopExit as CanonicalizeLoopExit, DeadBranchPrune as DeadBranchPrune, ExtractByteCode as ExtractByteCode, FindLiterallyCalls as FindLiterallyCalls, FixupArgs as FixupArgs, GenericRewrites as GenericRewrites, IRProcessing as IRProcessing, InlineClosureLikes as InlineClosureLikes, InlineInlinables as InlineInlinables, LiteralPropagationSubPipelinePass as LiteralPropagationSubPipelinePass, LiteralUnroll as LiteralUnroll, MakeFunctionToJitFunction as MakeFunctionToJitFunction, ReconstructSSA as ReconstructSSA, RewriteDynamicRaises as RewriteDynamicRaises, RewriteSemanticConstants as RewriteSemanticConstants, TranslateByteCode as TranslateByteCode, WithLifting as WithLifting
from numba.parfors.parfor import ParforDiagnostics as ParforDiagnostics
from typing import NamedTuple

class Flags(TargetConfig):
    __slots__: Incomplete
    enable_looplift: Incomplete
    enable_pyobject: Incomplete
    enable_pyobject_looplift: Incomplete
    enable_ssa: Incomplete
    force_pyobject: Incomplete
    release_gil: Incomplete
    no_compile: Incomplete
    debuginfo: Incomplete
    boundscheck: Incomplete
    forceinline: Incomplete
    no_cpython_wrapper: Incomplete
    no_cfunc_wrapper: Incomplete
    auto_parallel: Incomplete
    nrt: Incomplete
    no_rewrites: Incomplete
    error_model: Incomplete
    fastmath: Incomplete
    noalias: Incomplete
    inline: Incomplete
    dbg_extend_lifetimes: Incomplete
    dbg_optnone: Incomplete
    dbg_directives_only: Incomplete

DEFAULT_FLAGS: Incomplete
CR_FIELDS: Incomplete

class CompileResult(Incomplete):
    """
    A structure holding results from the compilation of a function.
    """
    __slots__: Incomplete
    def _reduce(self):
        """
        Reduce a CompileResult to picklable components.
        """
    def _find_referenced_environments(self):
        """Returns a list of referenced environments
        """
    @classmethod
    def _rebuild(cls, target_context, libdata, fndesc, env, signature, objectmode, lifted, typeann, reload_init, referenced_envs): ...
    @property
    def codegen(self): ...
    def dump(self, tab: str = '') -> None: ...

class _LowerResult(NamedTuple):
    fndesc: Incomplete
    call_helper: Incomplete
    cfunc: Incomplete
    env: Incomplete

def sanitize_compile_result_entries(entries): ...
def compile_result(**entries): ...
def run_frontend(func, inline_closures: bool = False, emit_dels: bool = False):
    """
    Run the compiler frontend over the given Python function, and return
    the function's canonical Numba IR.

    If inline_closures is Truthy then closure inlining will be run
    If emit_dels is Truthy the ir.Del nodes will be emitted appropriately
    """

class _CompileStatus:
    """
    Describes the state of compilation. Used like a C record.
    """
    __slots__: Incomplete
    fail_reason: Incomplete
    can_fallback: Incomplete
    def __init__(self, can_fallback) -> None: ...
    def __repr__(self) -> str: ...

class _EarlyPipelineCompletion(Exception):
    """
    Raised to indicate that a pipeline has completed early
    """
    result: Incomplete
    def __init__(self, result) -> None: ...

class StateDict(dict):
    """
    A dictionary that has an overloaded getattr and setattr to permit getting
    and setting key/values through the use of attributes.
    """
    def __getattr__(self, attr): ...
    def __setattr__(self, attr, value) -> None: ...

def _make_subtarget(targetctx, flags):
    """
    Make a new target context from the given target context and flags.
    """

class CompilerBase:
    """
    Stores and manages states for the compiler
    """
    state: Incomplete
    def __init__(self, typingctx, targetctx, library, args, return_type, flags, locals) -> None: ...
    def compile_extra(self, func): ...
    def compile_ir(self, func_ir, lifted=(), lifted_from: Incomplete | None = None): ...
    def define_pipelines(self) -> None:
        """Child classes override this to customize the pipelines in use.
        """
    def _compile_core(self):
        """
        Populate and run compiler pipeline
        """
    def _compile_bytecode(self):
        """
        Populate and run pipeline for bytecode input
        """
    def _compile_ir(self):
        """
        Populate and run pipeline for IR input
        """

class Compiler(CompilerBase):
    """The default compiler
    """
    def define_pipelines(self): ...

class DefaultPassBuilder:
    '''
    This is the default pass builder, it contains the "classic" default
    pipelines as pre-canned PassManager instances:
      - nopython
      - objectmode
      - interpreted
      - typed
      - untyped
      - nopython lowering
    '''
    @staticmethod
    def define_nopython_pipeline(state, name: str = 'nopython'):
        """Returns an nopython mode pipeline based PassManager
        """
    @staticmethod
    def define_nopython_lowering_pipeline(state, name: str = 'nopython_lowering'): ...
    @staticmethod
    def define_parfor_gufunc_nopython_lowering_pipeline(state, name: str = 'parfor_gufunc_nopython_lowering'): ...
    @staticmethod
    def define_typed_pipeline(state, name: str = 'typed'):
        """Returns the typed part of the nopython pipeline"""
    @staticmethod
    def define_parfor_gufunc_pipeline(state, name: str = 'parfor_gufunc_typed'):
        """Returns the typed part of the nopython pipeline"""
    @staticmethod
    def define_untyped_pipeline(state, name: str = 'untyped'):
        """Returns an untyped part of the nopython pipeline"""
    @staticmethod
    def define_objectmode_pipeline(state, name: str = 'object'):
        """Returns an object-mode pipeline based PassManager
        """

def compile_extra(typingctx, targetctx, func, args, return_type, flags, locals, library: Incomplete | None = None, pipeline_class=...):
    """Compiler entry point

    Parameter
    ---------
    typingctx :
        typing context
    targetctx :
        target context
    func : function
        the python function to be compiled
    args : tuple, list
        argument types
    return_type :
        Use ``None`` to indicate void return
    flags : numba.compiler.Flags
        compiler flags
    library : numba.codegen.CodeLibrary
        Used to store the compiled code.
        If it is ``None``, a new CodeLibrary is used.
    pipeline_class : type like numba.compiler.CompilerBase
        compiler pipeline
    """
def compile_ir(typingctx, targetctx, func_ir, args, return_type, flags, locals, lifted=(), lifted_from: Incomplete | None = None, is_lifted_loop: bool = False, library: Incomplete | None = None, pipeline_class=...):
    """
    Compile a function with the given IR.

    For internal use only.
    """
def compile_internal(typingctx, targetctx, library, func, args, return_type, flags, locals):
    """
    For internal use only.
    """
