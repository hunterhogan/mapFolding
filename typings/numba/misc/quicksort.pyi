from _typeshed import Incomplete
from numba.core import config as config, types as types
from typing import NamedTuple

class QuicksortImplementation(NamedTuple):
    compile: Incomplete
    partition: Incomplete
    partition3: Incomplete
    insertion_sort: Incomplete
    run_quicksort: Incomplete

class Partition(NamedTuple):
    start: Incomplete
    stop: Incomplete

SMALL_QUICKSORT: int
MAX_STACK: int

def make_quicksort_impl(wrap, lt: Incomplete | None = None, is_argsort: bool = False, is_list: bool = False, is_np_array: bool = False): ...
def make_py_quicksort(*args, **kwargs): ...
def make_jit_quicksort(*args, **kwargs): ...
