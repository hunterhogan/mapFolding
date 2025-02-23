from _typeshed import Incomplete

def _encode_string(s): ...
def _decode_string(b): ...

_shutting_down: Incomplete

def _at_shutdown() -> None: ...
def _is_shutting_down(_shutting_down=...):
    """
    Whether the interpreter is currently shutting down.
    For use in finalizers, __del__ methods, and similar; it is advised
    to early bind this function rather than look it up when calling it,
    since at shutdown module globals may be cleared.
    """
