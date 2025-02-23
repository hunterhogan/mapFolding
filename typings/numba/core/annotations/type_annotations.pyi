from _typeshed import Incomplete
from collections.abc import Mapping
from numba.core import ir as ir

class SourceLines(Mapping):
    lines: Incomplete
    startno: int
    def __init__(self, func) -> None: ...
    def __getitem__(self, lineno): ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    @property
    def avail(self): ...

class TypeAnnotation:
    func_data: Incomplete
    func_id: Incomplete
    blocks: Incomplete
    typemap: Incomplete
    calltypes: Incomplete
    filename: Incomplete
    linenum: Incomplete
    signature: Incomplete
    lifted: Incomplete
    num_lifted_loops: Incomplete
    lifted_from: Incomplete
    def __init__(self, func_ir, typemap, calltypes, lifted, lifted_from, args, return_type, html_output: Incomplete | None = None) -> None: ...
    def prepare_annotations(self): ...
    def annotate(self): ...
    def html_annotate(self, outfile) -> None: ...
    def annotate_raw(self):
        '''
        This returns "raw" annotation information i.e. it has no output format
        specific markup included.
        '''
    def __str__(self) -> str: ...

re_longest_white_prefix: Incomplete

def _getindent(text): ...
