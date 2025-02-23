from _typeshed import Incomplete
from numba.core import utils as utils

class Option:
    """An option to be used in ``TargetConfig``.
    """
    __slots__: Incomplete
    _type: Incomplete
    _default: Incomplete
    _doc: Incomplete
    def __init__(self, type, *, default, doc) -> None:
        """
        Parameters
        ----------
        type :
            Type of the option value. It can be a callable.
            The setter always calls ``self._type(value)``.
        default :
            The default value for the option.
        doc : str
            Docstring for the option.
        """
    @property
    def type(self): ...
    @property
    def default(self): ...
    @property
    def doc(self): ...

class _FlagsStack(utils.ThreadLocalStack, stack_name='flags'): ...

class ConfigStack:
    """A stack for tracking target configurations in the compiler.

    It stores the stack in a thread-local class attribute. All instances in the
    same thread will see the same stack.
    """
    @classmethod
    def top_or_none(cls):
        """Get the TOS or return None if no config is set.
        """
    _stk: Incomplete
    def __init__(self) -> None: ...
    def top(self): ...
    def __len__(self) -> int: ...
    def enter(self, flags):
        """Returns a contextmanager that performs ``push(flags)`` on enter and
        ``pop()`` on exit.
        """

class _MetaTargetConfig(type):
    """Metaclass for ``TargetConfig``.

    When a subclass of ``TargetConfig`` is created, all ``Option`` defined
    as class members will be parsed and corresponding getters, setters, and
    delters will be inserted.
    """
    def __init__(cls, name, bases, dct) -> None:
        """Invoked when subclass is created.

        Insert properties for each ``Option`` that are class members.
        All the options will be grouped inside the ``.options`` class
        attribute.
        """
    def find_options(cls, dct):
        """Returns a new dict with all the items that are a mapping to an
        ``Option``.
        """

class _NotSetType:
    def __repr__(self) -> str: ...

_NotSet: Incomplete

class TargetConfig(metaclass=_MetaTargetConfig):
    '''Base class for ``TargetConfig``.

    Subclass should fill class members with ``Option``. For example:

    >>> class MyTargetConfig(TargetConfig):
    >>>     a_bool_option = Option(type=bool, default=False, doc="a bool")
    >>>     an_int_option = Option(type=int, default=0, doc="an int")

    The metaclass will insert properties for each ``Option``. For example:

    >>> tc = MyTargetConfig()
    >>> tc.a_bool_option = True  # invokes the setter
    >>> print(tc.an_int_option)  # print the default
    '''
    __slots__: Incomplete
    _ZLIB_CONFIG: Incomplete
    _values: Incomplete
    def __init__(self, copy_from: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        copy_from : TargetConfig or None
            if None, creates an empty ``TargetConfig``.
            Otherwise, creates a copy.
        """
    def __repr__(self) -> str: ...
    def __hash__(self): ...
    def __eq__(self, other): ...
    def values(self):
        """Returns a dict of all the values
        """
    def is_set(self, name):
        """Is the option set?
        """
    def discard(self, name) -> None:
        """Remove the option by name if it is defined.

        After this, the value for the option will be set to its default value.
        """
    def inherit_if_not_set(self, name, default=...) -> None:
        """Inherit flag from ``ConfigStack``.

        Parameters
        ----------
        name : str
            Option name.
        default : optional
            When given, it overrides the default value.
            It is only used when the flag is not defined locally and there is
            no entry in the ``ConfigStack``.
        """
    def copy(self):
        """Clone this instance.
        """
    def summary(self) -> str:
        """Returns a ``str`` that summarizes this instance.

        In contrast to ``__repr__``, only options that are explicitly set will
        be shown.
        """
    def _guard_option(self, name) -> None: ...
    def _summary_args(self):
        """returns a sorted sequence of 2-tuple containing the
        ``(flag_name, flag_value)`` for flag that are set with a non-default
        value.
        """
    @classmethod
    def _make_compression_dictionary(cls) -> bytes:
        """Returns a ``bytes`` object suitable for use as a dictionary for
        compression.
        """
    def get_mangle_string(self) -> str:
        """Return a string suitable for symbol mangling.
        """
    @classmethod
    def demangle(cls, mangled: str) -> str:
        """Returns the demangled result from ``.get_mangle_string()``
        """
