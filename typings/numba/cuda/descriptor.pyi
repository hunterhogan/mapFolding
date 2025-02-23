from .target import CUDATargetContext as CUDATargetContext, CUDATypingContext as CUDATypingContext
from _typeshed import Incomplete
from numba.core.descriptors import TargetDescriptor as TargetDescriptor
from numba.core.options import TargetOptions as TargetOptions

class CUDATargetOptions(TargetOptions): ...

class CUDATarget(TargetDescriptor):
    options: Incomplete
    _typingctx: Incomplete
    _targetctx: Incomplete
    def __init__(self, name) -> None: ...
    @property
    def typing_context(self): ...
    @property
    def target_context(self): ...

cuda_target: Incomplete
