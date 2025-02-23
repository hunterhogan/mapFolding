def _inlining_threshold(optlevel, sizelevel: int = 0):
    """
    Compute the inlining threshold for the desired optimisation level

    Refer to http://llvm.org/docs/doxygen/html/InlineSimple_8cpp_source.html
    """
def create_pass_manager_builder(opt: int = 2, loop_vectorize: bool = False, slp_vectorize: bool = False):
    """
    Create an LLVM pass manager with the desired optimisation level and options.
    """
