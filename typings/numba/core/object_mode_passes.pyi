from numba.core import config as config, errors as errors, funcdesc as funcdesc, pylowering as pylowering, transforms as transforms, types as types, typing as typing
from numba.core.compiler_machinery import FunctionPass as FunctionPass, LoweringPass as LoweringPass, register_pass as register_pass

class ObjectModeFrontEnd(FunctionPass):
    _name: str
    def __init__(self) -> None: ...
    def _frontend_looplift(self, state):
        """
        Loop lifting analysis and transformation
        """
    def run_pass(self, state): ...

class ObjectModeBackEnd(LoweringPass):
    _name: str
    def __init__(self) -> None: ...
    def _py_lowering_stage(self, targetctx, library, interp, flags): ...
    def run_pass(self, state):
        """
        Lowering for object mode
        """
