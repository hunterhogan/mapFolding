from _typeshed import Incomplete
from numba.core import cgutils as cgutils, config as config, types as types
from numba.core.funcdesc import FunctionDescriptor as FunctionDescriptor

class GeneratorDescriptor(FunctionDescriptor):
    """
    The descriptor for a generator's next function.
    """
    __slots__: Incomplete
    @classmethod
    def from_generator_fndesc(cls, func_ir, fndesc, gentype, mangler):
        """
        Build a GeneratorDescriptor for the generator returned by the
        function described by *fndesc*, with type *gentype*.

        The generator inherits the env_name from the *fndesc*.
        All emitted functions for the generator shares the same Env.
        """
    @property
    def llvm_finalizer_name(self):
        """
        The LLVM name of the generator's finalizer function
        (if <generator type>.has_finalizer is true).
        """

class BaseGeneratorLower:
    """
    Base support class for lowering generators.
    """
    context: Incomplete
    fndesc: Incomplete
    library: Incomplete
    func_ir: Incomplete
    lower: Incomplete
    geninfo: Incomplete
    gentype: Incomplete
    gendesc: Incomplete
    arg_packer: Incomplete
    resume_blocks: Incomplete
    def __init__(self, lower) -> None: ...
    @property
    def call_conv(self): ...
    def get_args_ptr(self, builder, genptr): ...
    def get_resume_index_ptr(self, builder, genptr): ...
    def get_state_ptr(self, builder, genptr): ...
    def lower_init_func(self, lower) -> None:
        """
        Lower the generator's initialization function (which will fill up
        the passed-by-reference generator structure).
        """
    resume_index_ptr: Incomplete
    gen_state_ptr: Incomplete
    def lower_next_func(self, lower) -> None:
        """
        Lower the generator's next() function (which takes the
        passed-by-reference generator structure and returns the next
        yielded value).
        """
    def lower_finalize_func(self, lower) -> None:
        """
        Lower the generator's finalizer.
        """
    def return_from_generator(self, lower) -> None:
        """
        Emit a StopIteration at generator end and mark the generator exhausted.
        """
    def create_resumption_block(self, lower, index) -> None: ...
    def debug_print(self, builder, msg) -> None: ...

class GeneratorLower(BaseGeneratorLower):
    """
    Support class for lowering nopython generators.
    """
    def get_generator_type(self): ...
    def box_generator_struct(self, lower, gen_struct): ...
    def lower_finalize_func_body(self, builder, genptr) -> None:
        """
        Lower the body of the generator's finalizer: decref all live
        state variables.
        """

class PyGeneratorLower(BaseGeneratorLower):
    """
    Support class for lowering object mode generators.
    """
    def get_generator_type(self):
        '''
        Compute the actual generator type (the generator function\'s return
        type is simply "pyobject").
        '''
    def box_generator_struct(self, lower, gen_struct):
        """
        Box the raw *gen_struct* as a Python object.
        """
    def init_generator_state(self, lower) -> None:
        """
        NULL-initialize all generator state variables, to avoid spurious
        decref's on cleanup.
        """
    def lower_finalize_func_body(self, builder, genptr) -> None:
        """
        Lower the body of the generator's finalizer: decref all live
        state variables.
        """

class LowerYield:
    """
    Support class for lowering a particular yield point.
    """
    lower: Incomplete
    context: Incomplete
    builder: Incomplete
    genlower: Incomplete
    gentype: Incomplete
    gen_state_ptr: Incomplete
    resume_index_ptr: Incomplete
    yp: Incomplete
    inst: Incomplete
    live_vars: Incomplete
    live_var_indices: Incomplete
    def __init__(self, lower, yield_point, live_vars) -> None: ...
    def lower_yield_suspend(self) -> None: ...
    def lower_yield_resume(self) -> None: ...
