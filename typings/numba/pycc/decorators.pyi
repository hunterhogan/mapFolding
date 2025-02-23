from _typeshed import Incomplete
from numba.core import sigutils as sigutils, typing as typing
from numba.pycc.compiler import ExportEntry as ExportEntry

export_registry: Incomplete

def export(prototype): ...
def exportmany(prototypes): ...
def process_input_files(inputs) -> None:
    """
    Read input source files for execution of legacy @export / @exportmany
    decorators.
    """
def clear_export_registry() -> None: ...

re_symbol: Incomplete

def parse_prototype(text):
    '''Separate the symbol and function-type in a a string with
    "symbol function-type" (e.g. "mult float(float, float)")

    Returns
    ---------
    (symbol_string, functype_string)
    '''
