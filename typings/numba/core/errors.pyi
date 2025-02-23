import abc
from _typeshed import Incomplete
from abc import abstractmethod
from collections.abc import Generator

__all__ = ['NumbaWarning', 'NumbaPerformanceWarning', 'NumbaDeprecationWarning', 'NumbaPendingDeprecationWarning', 'NumbaParallelSafetyWarning', 'NumbaTypeSafetyWarning', 'NumbaExperimentalFeatureWarning', 'NumbaInvalidConfigWarning', 'NumbaPedanticWarning', 'NumbaIRAssumptionWarning', 'NumbaDebugInfoWarning', 'NumbaSystemWarning', 'NumbaError', 'UnsupportedError', 'UnsupportedBytecodeError', 'UnsupportedRewriteError', 'IRError', 'RedefinedError', 'NotDefinedError', 'VerificationError', 'DeprecationError', 'LoweringError', 'UnsupportedParforsError', 'ForbiddenConstruct', 'TypingError', 'UntypedAttributeError', 'ByteCodeSupportError', 'CompilerError', 'ConstantInferenceError', 'InternalError', 'InternalTargetMismatchError', 'NonexistentTargetError', 'RequireLiteralValue', 'ForceLiteralArg', 'LiteralTypingError', 'NumbaValueError', 'NumbaTypeError', 'NumbaAttributeError', 'NumbaAssertionError', 'NumbaNotImplementedError', 'NumbaKeyError', 'NumbaIndexError', 'NumbaRuntimeError']

class NumbaWarning(Warning):
    """
    Base category for all Numba compiler warnings.
    """
    msg: Incomplete
    loc: Incomplete
    def __init__(self, msg, loc: Incomplete | None = None, highlighting: bool = True) -> None: ...

class NumbaPerformanceWarning(NumbaWarning):
    """
    Warning category for when an operation might not be
    as fast as expected.
    """
class NumbaDeprecationWarning(NumbaWarning, DeprecationWarning):
    """
    Warning category for use of a deprecated feature.
    """
class NumbaPendingDeprecationWarning(NumbaWarning, PendingDeprecationWarning):
    """
    Warning category for use of a feature that is pending deprecation.
    """
class NumbaParallelSafetyWarning(NumbaWarning):
    """
    Warning category for when an operation in a prange
    might not have parallel semantics.
    """
class NumbaTypeSafetyWarning(NumbaWarning):
    """
    Warning category for unsafe casting operations.
    """
class NumbaExperimentalFeatureWarning(NumbaWarning):
    """
    Warning category for using an experimental feature.
    """
class NumbaInvalidConfigWarning(NumbaWarning):
    """
    Warning category for using an invalid configuration.
    """

class NumbaPedanticWarning(NumbaWarning):
    """
    Warning category for reporting pedantic messages.
    """
    def __init__(self, msg, **kwargs) -> None: ...

class NumbaIRAssumptionWarning(NumbaPedanticWarning):
    """
    Warning category for reporting an IR assumption violation.
    """
class NumbaDebugInfoWarning(NumbaWarning):
    """
    Warning category for an issue with the emission of debug information.
    """
class NumbaSystemWarning(NumbaWarning):
    """
    Warning category for an issue with the system configuration.
    """

class _ColorScheme(metaclass=abc.ABCMeta):
    @abstractmethod
    def code(self, msg): ...
    @abstractmethod
    def errmsg(self, msg): ...
    @abstractmethod
    def filename(self, msg): ...
    @abstractmethod
    def indicate(self, msg): ...
    @abstractmethod
    def highlight(self, msg): ...
    @abstractmethod
    def reset(self, msg): ...

class _DummyColorScheme(_ColorScheme):
    def __init__(self, theme: Incomplete | None = None) -> None: ...
    def code(self, msg) -> None: ...
    def errmsg(self, msg) -> None: ...
    def filename(self, msg) -> None: ...
    def indicate(self, msg) -> None: ...
    def highlight(self, msg) -> None: ...
    def reset(self, msg) -> None: ...

class NOPColorScheme(_DummyColorScheme):
    def __init__(self, theme: Incomplete | None = None) -> None: ...
    def code(self, msg): ...
    def errmsg(self, msg): ...
    def filename(self, msg): ...
    def indicate(self, msg): ...
    def highlight(self, msg): ...
    def reset(self, msg): ...

class ColorShell:
    _has_initialized: bool
    def __init__(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *exc_detail) -> None: ...

class reset_terminal:
    _buf: Incomplete
    def __init__(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *exc_detail) -> None: ...

class HighlightColorScheme(_DummyColorScheme):
    _code: Incomplete
    _errmsg: Incomplete
    _filename: Incomplete
    _indicate: Incomplete
    _highlight: Incomplete
    _reset: Incomplete
    def __init__(self, theme=...) -> None: ...
    def _markup(self, msg, color: Incomplete | None = None, style=...): ...
    def code(self, msg): ...
    def errmsg(self, msg): ...
    def filename(self, msg): ...
    def indicate(self, msg): ...
    def highlight(self, msg): ...
    def reset(self, msg): ...

class WarningsFixer:
    '''
    An object "fixing" warnings of a given category caught during
    certain phases.  The warnings can have their filename and lineno fixed,
    and they are deduplicated as well.

    When used as a context manager, any warnings caught by `.catch_warnings()`
    will be flushed at the exit of the context manager.
    '''
    _category: Incomplete
    _warnings: Incomplete
    def __init__(self, category) -> None: ...
    def catch_warnings(self, filename: Incomplete | None = None, lineno: Incomplete | None = None) -> Generator[None]:
        """
        Store warnings and optionally fix their filename and lineno.
        """
    def flush(self):
        """
        Emit all stored warnings.
        """
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...

class NumbaError(Exception):
    msg: Incomplete
    loc: Incomplete
    def __init__(self, msg, loc: Incomplete | None = None, highlighting: bool = True) -> None: ...
    _contexts: Incomplete
    @property
    def contexts(self): ...
    args: Incomplete
    def add_context(self, msg):
        """
        Add contextual info.  The exception message is expanded with the new
        contextual information.
        """
    def patch_message(self, new_message) -> None:
        """
        Change the error message to the given new message.
        """

class UnsupportedError(NumbaError):
    """
    Numba does not have an implementation for this functionality.
    """

class UnsupportedBytecodeError(Exception):
    """Unsupported bytecode is non-recoverable
    """
    def __init__(self, msg, loc: Incomplete | None = None) -> None: ...

class UnsupportedRewriteError(UnsupportedError):
    """UnsupportedError from rewrite passes
    """
class IRError(NumbaError):
    """
    An error occurred during Numba IR generation.
    """
class RedefinedError(IRError):
    """
    An error occurred during interpretation of IR due to variable redefinition.
    """

class NotDefinedError(IRError):
    """
    An undefined variable is encountered during interpretation of IR.
    """
    name: Incomplete
    def __init__(self, name, loc: Incomplete | None = None) -> None: ...

class VerificationError(IRError):
    """
    An error occurred during IR verification. Once Numba's internal
    representation (IR) is constructed it is then verified to ensure that
    terminators are both present and in the correct places within the IR. If
    it is the case that this condition is not met, a VerificationError is
    raised.
    """
class DeprecationError(NumbaError):
    """
    Functionality is deprecated.
    """

class LoweringError(NumbaError):
    """
    An error occurred during lowering.
    """
    def __init__(self, msg, loc: Incomplete | None = None) -> None: ...

class UnsupportedParforsError(NumbaError):
    """
    An error occurred because parfors is not supported on the platform.
    """
class ForbiddenConstruct(LoweringError):
    """
    A forbidden Python construct was encountered (e.g. use of locals()).
    """
class TypingError(NumbaError):
    """
    A type inference failure.
    """

class UntypedAttributeError(TypingError):
    def __init__(self, value, attr, loc: Incomplete | None = None) -> None: ...

class ByteCodeSupportError(NumbaError):
    """
    Failure to extract the bytecode of the user's function.
    """
    def __init__(self, msg, loc: Incomplete | None = None) -> None: ...

class CompilerError(NumbaError):
    """
    Some high-level error in the compiler.
    """

class ConstantInferenceError(NumbaError):
    """
    Failure during constant inference.
    """
    def __init__(self, value, loc: Incomplete | None = None) -> None: ...

class InternalError(NumbaError):
    """
    For wrapping internal error occurred within the compiler
    """
    old_exception: Incomplete
    def __init__(self, exception) -> None: ...

class InternalTargetMismatchError(InternalError):
    """For signalling a target mismatch error occurred internally within the
    compiler.
    """
    def __init__(self, kind, target_hw, hw_clazz) -> None: ...

class NonexistentTargetError(InternalError):
    """For signalling that a target that does not exist was requested.
    """
class RequireLiteralValue(TypingError):
    """
    For signalling that a function's typing requires a constant value for
    some of its arguments.
    """

class ForceLiteralArg(NumbaError):
    """A Pseudo-exception to signal the dispatcher to type an argument literally

    Attributes
    ----------
    requested_args : frozenset[int]
        requested positions of the arguments.
    """
    requested_args: Incomplete
    fold_arguments: Incomplete
    def __init__(self, arg_indices, fold_arguments: Incomplete | None = None, loc: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        arg_indices : Sequence[int]
            requested positions of the arguments.
        fold_arguments: callable
            A function ``(tuple, dict) -> tuple`` that binds and flattens
            the ``args`` and ``kwargs``.
        loc : numba.ir.Loc or None
        """
    def bind_fold_arguments(self, fold_arguments):
        """Bind the fold_arguments function
        """
    def combine(self, other):
        """Returns a new instance by or'ing the requested_args.
        """
    def __or__(self, other):
        """Same as self.combine(other)
        """

class LiteralTypingError(TypingError):
    """
    Failure in typing a Literal type
    """
class NumbaValueError(TypingError): ...
class NumbaTypeError(TypingError): ...
class NumbaAttributeError(TypingError): ...
class NumbaAssertionError(TypingError): ...
class NumbaNotImplementedError(TypingError): ...
class NumbaKeyError(TypingError): ...
class NumbaIndexError(TypingError): ...
class NumbaRuntimeError(NumbaError): ...
