import threading
from _typeshed import Incomplete
from numba.core import config as config

class TLS(threading.local):
    """Use a subclass to properly initialize the TLS variables in all threads."""
    tracing: bool
    indent: int
    def __init__(self) -> None: ...

tls: Incomplete

def find_function_info(func, spec, args):
    """Return function meta-data in a tuple.

    (name, type)"""
def chop(value): ...
def create_events(fname, spec, args, kwds): ...
def dotrace(*args, **kwds):
    """Function decorator to trace a function's entry and exit.

    *args: categories in which to trace this function. Example usage:

    @trace
    def function(...):...

    @trace('mycategory')
    def function(...):...


    """
def notrace(*args, **kwds):
    """Just a no-op in case tracing is disabled."""
def doevent(msg) -> None: ...
def noevent(msg) -> None: ...

logger: Incomplete
trace = dotrace
event = doevent
trace = notrace
event = noevent
