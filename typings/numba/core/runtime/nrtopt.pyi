from _typeshed import Incomplete
from numba.core import cgutils as cgutils

_regex_incref: Incomplete
_regex_decref: Incomplete
_regex_bb: Incomplete

def _remove_redundant_nrt_refct(llvmir): ...
def remove_redundant_nrt_refct(ll_module):
    """
    Remove redundant reference count operations from the
    `llvmlite.binding.ModuleRef`. This parses the ll_module as a string and
    line by line to remove the unnecessary nrt refct pairs within each block.
    Decref calls are moved after the last incref call in the block to avoid
    temporarily decref'ing to zero (which can happen due to hidden decref from
    alias).

    Note: non-threadsafe due to usage of global LLVMcontext
    """
