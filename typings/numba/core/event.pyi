import abc
import enum
from _typeshed import Incomplete
from collections.abc import Generator
from numba.core import config as config, utils as utils

class EventStatus(enum.Enum):
    """Status of an event.
    """
    START = ...
    END = ...

_builtin_kinds: Incomplete

def _guard_kind(kind):
    '''Guard to ensure that an event kind is valid.

    All event kinds with a "numba:" prefix must be defined in the pre-defined
    ``numba.core.event._builtin_kinds``.
    Custom event kinds are allowed by not using the above prefix.

    Parameters
    ----------
    kind : str

    Return
    ------
    res : str
    '''

class Event:
    """An event.

    Parameters
    ----------
    kind : str
    status : EventStatus
    data : any; optional
        Additional data for the event.
    exc_details : 3-tuple; optional
        Same 3-tuple for ``__exit__``.
    """
    _kind: Incomplete
    _status: Incomplete
    _data: Incomplete
    _exc_details: Incomplete
    def __init__(self, kind, status, data: Incomplete | None = None, exc_details: Incomplete | None = None) -> None: ...
    @property
    def kind(self):
        """Event kind

        Returns
        -------
        res : str
        """
    @property
    def status(self):
        """Event status

        Returns
        -------
        res : EventStatus
        """
    @property
    def data(self):
        """Event data

        Returns
        -------
        res : object
        """
    @property
    def is_start(self):
        """Is it a *START* event?

        Returns
        -------
        res : bool
        """
    @property
    def is_end(self):
        """Is it an *END* event?

        Returns
        -------
        res : bool
        """
    @property
    def is_failed(self):
        """Is the event carrying an exception?

        This is used for *END* event. This method will never return ``True``
        in a *START* event.

        Returns
        -------
        res : bool
        """
    def __str__(self) -> str: ...
    __repr__ = __str__

_registered: Incomplete

def register(kind, listener) -> None:
    """Register a listener for a given event kind.

    Parameters
    ----------
    kind : str
    listener : Listener
    """
def unregister(kind, listener) -> None:
    """Unregister a listener for a given event kind.

    Parameters
    ----------
    kind : str
    listener : Listener
    """
def broadcast(event) -> None:
    """Broadcast an event to all registered listeners.

    Parameters
    ----------
    event : Event
    """

class Listener(abc.ABC, metaclass=abc.ABCMeta):
    """Base class for all event listeners.
    """
    @abc.abstractmethod
    def on_start(self, event):
        """Called when there is a *START* event.

        Parameters
        ----------
        event : Event
        """
    @abc.abstractmethod
    def on_end(self, event):
        """Called when there is a *END* event.

        Parameters
        ----------
        event : Event
        """
    def notify(self, event) -> None:
        """Notify this Listener with the given Event.

        Parameters
        ----------
        event : Event
        """

class TimingListener(Listener):
    """A listener that measures the total time spent between *START* and
    *END* events during the time this listener is active.
    """
    _depth: int
    def __init__(self) -> None: ...
    _ts: Incomplete
    def on_start(self, event) -> None: ...
    _duration: Incomplete
    def on_end(self, event) -> None: ...
    @property
    def done(self):
        """Returns a ``bool`` indicating whether a measurement has been made.

        When this returns ``False``, the matching event has never fired.
        If and only if this returns ``True``, ``.duration`` can be read without
        error.
        """
    @property
    def duration(self):
        """Returns the measured duration.

        This may raise ``AttributeError``. Users can use ``.done`` to check
        that a measurement has been made.
        """

class RecordingListener(Listener):
    """A listener that records all events and stores them in the ``.buffer``
    attribute as a list of 2-tuple ``(float, Event)``, where the first element
    is the time the event occurred as returned by ``time.time()`` and the second
    element is the event.
    """
    buffer: Incomplete
    def __init__(self) -> None: ...
    def on_start(self, event) -> None: ...
    def on_end(self, event) -> None: ...

def install_listener(kind, listener) -> Generator[Incomplete]:
    '''Install a listener for event "kind" temporarily within the duration of
    the context.

    Returns
    -------
    res : Listener
        The *listener* provided.

    Examples
    --------

    >>> with install_listener("numba:compile", listener):
    >>>     some_code()  # listener will be active here.
    >>> other_code()     # listener will be unregistered by this point.

    '''
def install_timer(kind, callback) -> Generator[Incomplete]:
    """Install a TimingListener temporarily to measure the duration of
    an event.

    If the context completes successfully, the *callback* function is executed.
    The *callback* function is expected to take a float argument for the
    duration in seconds.

    Returns
    -------
    res : TimingListener

    Examples
    --------

    This is equivalent to:

    >>> with install_listener(kind, TimingListener()) as res:
    >>>    ...
    """
def install_recorder(kind) -> Generator[Incomplete]:
    """Install a RecordingListener temporarily to record all events.

    Once the context is closed, users can use ``RecordingListener.buffer``
    to access the recorded events.

    Returns
    -------
    res : RecordingListener

    Examples
    --------

    This is equivalent to:

    >>> with install_listener(kind, RecordingListener()) as res:
    >>>    ...
    """
def start_event(kind, data: Incomplete | None = None) -> None:
    """Trigger the start of an event of *kind* with *data*.

    Parameters
    ----------
    kind : str
        Event kind.
    data : any; optional
        Extra event data.
    """
def end_event(kind, data: Incomplete | None = None, exc_details: Incomplete | None = None) -> None:
    """Trigger the end of an event of *kind*, *exc_details*.

    Parameters
    ----------
    kind : str
        Event kind.
    data : any; optional
        Extra event data.
    exc_details : 3-tuple; optional
        Same 3-tuple for ``__exit__``. Or, ``None`` if no error.
    """
def trigger_event(kind, data: Incomplete | None = None) -> Generator[None]:
    """A context manager to trigger the start and end events of *kind* with
    *data*. The start event is triggered when entering the context.
    The end event is triggered when exiting the context.

    Parameters
    ----------
    kind : str
        Event kind.
    data : any; optional
        Extra event data.
    """
def _prepare_chrome_trace_data(listener: RecordingListener):
    """Prepare events in `listener` for serializing as chrome trace data.
    """
def _setup_chrome_trace_exit_handler() -> None:
    """Setup a RecordingListener and an exit handler to write the captured
    events to file.
    """
