import abc
from _typeshed import Incomplete
from collections.abc import Generator
from numba.core import config as config, errors as errors, funcdesc as funcdesc, ir as ir, lowering as lowering, postproc as postproc, rewrites as rewrites, typeinfer as typeinfer, types as types, typing as typing
from numba.core.annotations import type_annotations as type_annotations
from numba.core.compiler_machinery import AnalysisPass as AnalysisPass, FunctionPass as FunctionPass, LoweringPass as LoweringPass, register_pass as register_pass
from numba.core.ir_utils import build_definitions as build_definitions, check_and_legalize_ir as check_and_legalize_ir, compute_cfg_from_blocks as compute_cfg_from_blocks, dead_code_elimination as dead_code_elimination, get_definition as get_definition, guard as guard, is_operator_or_getitem as is_operator_or_getitem, raise_on_unsupported_feature as raise_on_unsupported_feature, replace_vars as replace_vars, simplify_CFG as simplify_CFG, warn_deprecated as warn_deprecated
from numba.parfors.parfor import Parfor as Parfor
from numba.parfors.parfor_lowering import ParforLower as ParforLower
from typing import NamedTuple

class _TypingResults(NamedTuple):
    typemap: Incomplete
    return_type: Incomplete
    calltypes: Incomplete
    typing_errors: Incomplete

def fallback_context(state, msg) -> Generator[None]:
    """
    Wraps code that would signal a fallback to object mode
    """
def type_inference_stage(typingctx, targetctx, interp, args, return_type, locals={}, raise_errors: bool = True): ...

class BaseTypeInference(FunctionPass):
    _raise_errors: bool
    def __init__(self) -> None: ...
    def run_pass(self, state):
        """
        Type inference and legalization
        """

class NopythonTypeInference(BaseTypeInference):
    _name: str

class PartialTypeInference(BaseTypeInference):
    _name: str
    _raise_errors: bool

class AnnotateTypes(AnalysisPass):
    _name: str
    def __init__(self) -> None: ...
    def get_analysis_usage(self, AU) -> None: ...
    def run_pass(self, state):
        """
        Create type annotation after type inference
        """

class NopythonRewrites(FunctionPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state):
        """
        Perform any intermediate representation rewrites after type
        inference.
        """

class PreParforPass(FunctionPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state):
        """
        Preprocessing for data-parallel computations.
        """

def _reload_parfors() -> None:
    """Reloader for cached parfors
    """

class ParforPass(FunctionPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state):
        """
        Convert data-parallel computations into Parfor nodes
        """

class ParforFusionPass(FunctionPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state):
        """
        Do fusion of parfor nodes.
        """

class ParforPreLoweringPass(FunctionPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state):
        """
        Prepare parfors for lowering.
        """

class DumpParforDiagnostics(AnalysisPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...

class BaseNativeLowering(abc.ABC, LoweringPass, metaclass=abc.ABCMeta):
    """The base class for a lowering pass. The lowering functionality must be
    specified in inheriting classes by providing an appropriate lowering class
    implementation in the overridden `lowering_class` property."""
    _name: Incomplete
    def __init__(self) -> None: ...
    @property
    @abc.abstractmethod
    def lowering_class(self):
        """Returns the class that performs the lowering of the IR describing the
        function that is the target of the current compilation."""
    def run_pass(self, state): ...

class NativeLowering(BaseNativeLowering):
    """Lowering pass for a native function IR described solely in terms of
     Numba's standard `numba.core.ir` nodes."""
    _name: str
    @property
    def lowering_class(self): ...

class NativeParforLowering(BaseNativeLowering):
    """Lowering pass for a native function IR described using Numba's standard
    `numba.core.ir` nodes and also parfor.Parfor nodes."""
    _name: str
    @property
    def lowering_class(self): ...

class NoPythonSupportedFeatureValidation(AnalysisPass):
    """NoPython Mode check: Validates the IR to ensure that features in use are
    in a form that is supported"""
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...

class IRLegalization(AnalysisPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...

class NoPythonBackend(LoweringPass):
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state):
        """
        Back-end: Generate LLVM IR from Numba IR, compile to machine code
        """

class InlineOverloads(FunctionPass):
    """
    This pass will inline a function wrapped by the numba.extending.overload
    decorator directly into the site of its call depending on the value set in
    the 'inline' kwarg to the decorator.

    This is a typed pass. CFG simplification and DCE are performed on
    completion.
    """
    _name: str
    def __init__(self) -> None: ...
    _DEBUG: bool
    def run_pass(self, state):
        """Run inlining of overloads
        """
    def _get_attr_info(self, state, expr): ...
    def _get_callable_info(self, state, expr): ...
    def _do_work_expr(self, state, work_list, block, i, expr, inline_worker): ...
    def _run_inliner(self, state, inline_type, sig, template, arg_typs, expr, i, impl, block, work_list, is_method, inline_worker): ...
    def _add_method_self_arg(self, state, expr): ...

class DeadCodeElimination(FunctionPass):
    """
    Does dead code elimination
    """
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...

class PreLowerStripPhis(FunctionPass):
    """Remove phi nodes (ir.Expr.phi) introduced by SSA.

    This is needed before Lowering because the phi nodes in Numba IR do not
    match the semantics of phi nodes in LLVM IR. In Numba IR, phi nodes may
    expand into multiple LLVM instructions.
    """
    _name: str
    def __init__(self) -> None: ...
    def run_pass(self, state): ...
    def _strip_phi_nodes(self, func_ir):
        """Strip Phi nodes from ``func_ir``

        For each phi node, put incoming value to their respective incoming
        basic-block at possibly the latest position (i.e. after the latest
        assignment to the corresponding variable).
        """
    def _simplify_conditionally_defined_variable(self, func_ir):
        """
        Rewrite assignments like:

            ver1 = null()
            ...
            ver1 = ver
            ...
            uses(ver1)

        into:
            # delete all assignments to ver1
            uses(ver)

        This is only needed for parfors because the SSA pass will create extra
        variable assignments that the parfor code does not expect.
        This pass helps avoid problems by reverting the effect of SSA.
        """
