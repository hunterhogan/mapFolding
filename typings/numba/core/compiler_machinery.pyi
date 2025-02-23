import abc
from _typeshed import Incomplete
from abc import ABCMeta, abstractmethod
from numba.core import config as config, errors as errors, transforms as transforms, utils as utils
from numba.core.compiler_lock import global_compiler_lock as global_compiler_lock
from numba.core.ir_utils import enforce_no_dels as enforce_no_dels, legalize_single_scope as legalize_single_scope
from numba.core.postproc import PostProcessor as PostProcessor
from numba.core.tracing import event as event
from typing import NamedTuple

_termcolor: Incomplete

class SimpleTimer:
    """
    A simple context managed timer
    """
    ts: Incomplete
    def __enter__(self): ...
    elapsed: Incomplete
    def __exit__(self, *exc) -> None: ...

class CompilerPass(metaclass=ABCMeta):
    """ The base class for all compiler passes.
    """
    _analysis: Incomplete
    _pass_id: Incomplete
    @abstractmethod
    def __init__(self, *args, **kwargs): ...
    @classmethod
    def name(cls):
        """
        Returns the name of the pass
        """
    @property
    def pass_id(self):
        """
        The ID of the pass
        """
    @pass_id.setter
    def pass_id(self, val) -> None:
        """
        Sets the ID of the pass
        """
    @property
    def analysis(self):
        """
        Analysis data for the pass
        """
    @analysis.setter
    def analysis(self, val) -> None:
        """
        Set the analysis data for the pass
        """
    def run_initialization(self, *args, **kwargs):
        """
        Runs the initialization sequence for the pass, will run before
        `run_pass`.
        """
    @abstractmethod
    def run_pass(self, *args, **kwargs):
        """
        Runs the pass itself. Must return True/False depending on whether
        statement level modification took place.
        """
    def run_finalizer(self, *args, **kwargs):
        """
        Runs the initialization sequence for the pass, will run before
        `run_pass`.
        """
    def get_analysis_usage(self, AU) -> None:
        """ Override to set analysis usage
        """
    def get_analysis(self, pass_name):
        """
        Gets the analysis from a given pass
        """

class SSACompliantMixin:
    """ Mixin to indicate a pass is SSA form compliant. Nothing is asserted
    about this condition at present.
    """
class FunctionPass(CompilerPass, metaclass=abc.ABCMeta):
    """ Base class for function passes
    """
class AnalysisPass(CompilerPass, metaclass=abc.ABCMeta):
    """ Base class for analysis passes (no modification made to state)
    """
class LoweringPass(CompilerPass, metaclass=abc.ABCMeta):
    """ Base class for lowering passes
    """

class AnalysisUsage:
    """This looks and behaves like LLVM's AnalysisUsage because its like that.
    """
    _required: Incomplete
    _preserved: Incomplete
    def __init__(self) -> None: ...
    def get_required_set(self): ...
    def get_preserved_set(self): ...
    def add_required(self, pss) -> None: ...
    def add_preserved(self, pss) -> None: ...
    def __str__(self) -> str: ...

_DEBUG: bool

def debug_print(*args, **kwargs) -> None: ...

class pass_timings(NamedTuple):
    init: Incomplete
    run: Incomplete
    finalize: Incomplete

class PassManager:
    """
    The PassManager is a named instance of a particular compilation pipeline
    """
    _ENFORCING: bool
    passes: Incomplete
    exec_times: Incomplete
    _finalized: bool
    _analysis: Incomplete
    _print_after: Incomplete
    pipeline_name: Incomplete
    def __init__(self, pipeline_name) -> None:
        '''
        Create a new pipeline with name "pipeline_name"
        '''
    def _validate_pass(self, pass_cls) -> None: ...
    def add_pass(self, pss, description: str = '') -> None:
        """
        Append a pass to the PassManager's compilation pipeline
        """
    def add_pass_after(self, pass_cls, location) -> None:
        """
        Add a pass `pass_cls` to the PassManager's compilation pipeline after
        the pass `location`.
        """
    def _debug_init(self): ...
    def finalize(self) -> None:
        """
        Finalize the PassManager, after which no more passes may be added
        without re-finalization.
        """
    @property
    def finalized(self): ...
    def _patch_error(self, desc, exc):
        """
        Patches the error to show the stage that it arose in.
        """
    def _runPass(self, index, pss, internal_state): ...
    def run(self, state) -> None:
        """
        Run the defined pipelines on the state.
        """
    def dependency_analysis(self):
        """
        Computes dependency analysis
        """

class pass_info(NamedTuple):
    pass_inst: Incomplete
    mutates_CFG: Incomplete
    analysis_only: Incomplete

class PassRegistry:
    """
    Pass registry singleton class.
    """
    _id: int
    _registry: Incomplete
    def register(self, mutates_CFG, analysis_only): ...
    def is_registered(self, clazz): ...
    def get(self, clazz): ...
    def _does_pass_name_alias(self, check): ...
    def find_by_name(self, class_name): ...
    def dump(self) -> None: ...

_pass_registry: Incomplete
register_pass: Incomplete
