from _typeshed import Incomplete
from typing import NamedTuple

SMALL_MERGESORT: int

class MergesortImplementation(NamedTuple):
    run_mergesort: Incomplete

def make_mergesort_impl(wrap, lt: Incomplete | None = None, is_argsort: bool = False): ...
def make_jit_mergesort(*args, **kwargs): ...
