from _typeshed import Incomplete
from numba.core import types as types
from typing import NamedTuple

class TimsortImplementation(NamedTuple):
    compile: Incomplete
    count_run: Incomplete
    binarysort: Incomplete
    gallop_left: Incomplete
    gallop_right: Incomplete
    merge_init: Incomplete
    merge_append: Incomplete
    merge_pop: Incomplete
    merge_compute_minrun: Incomplete
    merge_lo: Incomplete
    merge_hi: Incomplete
    merge_at: Incomplete
    merge_force_collapse: Incomplete
    merge_collapse: Incomplete
    run_timsort: Incomplete
    run_timsort_with_values: Incomplete

MAX_MERGE_PENDING: int
MIN_GALLOP: int
MERGESTATE_TEMP_SIZE: int

class MergeState(NamedTuple):
    min_gallop: Incomplete
    keys: Incomplete
    values: Incomplete
    pending: Incomplete
    n: Incomplete

class MergeRun(NamedTuple):
    start: Incomplete
    size: Incomplete

def make_timsort_impl(wrap, make_temp_area): ...
def make_py_timsort(*args): ...
def make_jit_timsort(*args): ...
