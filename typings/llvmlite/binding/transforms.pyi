from _typeshed import Incomplete
from llvmlite.binding import ffi as ffi, passmanagers as passmanagers

def create_pass_manager_builder(): ...

class PassManagerBuilder(ffi.ObjectRef):
    __slots__: Incomplete
    def __init__(self, ptr: Incomplete | None = None) -> None: ...
    @property
    def opt_level(self):
        """
        The general optimization level as an integer between 0 and 3.
        """
    @opt_level.setter
    def opt_level(self, level) -> None: ...
    @property
    def size_level(self):
        """
        Whether and how much to optimize for size.  An integer between 0 and 2.
        """
    @size_level.setter
    def size_level(self, size) -> None: ...
    @property
    def inlining_threshold(self) -> None:
        """
        The integer threshold for inlining a function into another.  The higher,
        the more likely inlining a function is.  This attribute is write-only.
        """
    @inlining_threshold.setter
    def inlining_threshold(self, threshold) -> None: ...
    @property
    def disable_unroll_loops(self):
        """
        If true, disable loop unrolling.
        """
    @disable_unroll_loops.setter
    def disable_unroll_loops(self, disable: bool = True) -> None: ...
    @property
    def loop_vectorize(self):
        """
        If true, allow vectorizing loops.
        """
    @loop_vectorize.setter
    def loop_vectorize(self, enable: bool = True): ...
    @property
    def slp_vectorize(self):
        '''
        If true, enable the "SLP vectorizer", which uses a different algorithm
        from the loop vectorizer.  Both may be enabled at the same time.
        '''
    @slp_vectorize.setter
    def slp_vectorize(self, enable: bool = True): ...
    def _populate_module_pm(self, pm) -> None: ...
    def _populate_function_pm(self, pm) -> None: ...
    def populate(self, pm) -> None: ...
    def _dispose(self) -> None: ...
