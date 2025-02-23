from _typeshed import Incomplete
from collections.abc import Generator, Sequence
from functools import cached_property as cached_property
from numba.core import config as config
from typing import NamedTuple

class RecordLLVMPassTimings:
    """A helper context manager to track LLVM pass timings.
    """
    __slots__: Incomplete
    def __enter__(self):
        """Enables the pass timing in LLVM.
        """
    _data: Incomplete
    def __exit__(self, exc_val: type[BaseException] | None, exc_type: BaseException | None, exc_tb: types.TracebackType | None) -> None:
        """Reset timings and save report internally.
        """
    def get(self):
        """Retrieve timing data for processing.

        Returns
        -------
        timings: ProcessedPassTimings
        """

class PassTimingRecord(NamedTuple):
    user_time: Incomplete
    user_percent: Incomplete
    system_time: Incomplete
    system_percent: Incomplete
    user_system_time: Incomplete
    user_system_percent: Incomplete
    wall_time: Incomplete
    wall_percent: Incomplete
    pass_name: Incomplete
    instruction: Incomplete

def _adjust_timings(records):
    """Adjust timing records because of truncated information.

    Details: The percent information can be used to improve the timing
    information.

    Returns
    -------
    res: List[PassTimingRecord]
    """

class ProcessedPassTimings:
    """A class for processing raw timing report from LLVM.

    The processing is done lazily so we don't waste time processing unused
    timing information.
    """
    _raw_data: Incomplete
    def __init__(self, raw_data) -> None: ...
    def __bool__(self) -> bool: ...
    def get_raw_data(self):
        """Returns the raw string data.

        Returns
        -------
        res: str
        """
    def get_total_time(self):
        """Compute the total time spend in all passes.

        Returns
        -------
        res: float
        """
    def list_records(self):
        """Get the processed data for the timing report.

        Returns
        -------
        res: List[PassTimingRecord]
        """
    def list_top(self, n):
        """Returns the top(n) most time-consuming (by wall-time) passes.

        Parameters
        ----------
        n: int
            This limits the maximum number of items to show.
            This function will show the ``n`` most time-consuming passes.

        Returns
        -------
        res: List[PassTimingRecord]
            Returns the top(n) most time-consuming passes in descending order.
        """
    def summary(self, topn: int = 5, indent: int = 0):
        """Return a string summarizing the timing information.

        Parameters
        ----------
        topn: int; optional
            This limits the maximum number of items to show.
            This function will show the ``topn`` most time-consuming passes.
        indent: int; optional
            Set the indentation level. Defaults to 0 for no indentation.

        Returns
        -------
        res: str
        """
    @cached_property
    def _processed(self):
        """A cached property for lazily processing the data and returning it.

        See ``_process()`` for details.
        """
    def _process(self):
        """Parses the raw string data from LLVM timing report and attempts
        to improve the data by recomputing the times
        (See `_adjust_timings()``).
        """

class NamedTimings(NamedTuple):
    name: Incomplete
    timings: Incomplete

class PassTimingsCollection(Sequence):
    """A collection of pass timings.

    This class implements the ``Sequence`` protocol for accessing the
    individual timing records.
    """
    _name: Incomplete
    _records: Incomplete
    def __init__(self, name) -> None: ...
    def record(self, name) -> Generator[None]:
        """Record new timings and append to this collection.

        Note: this is mainly for internal use inside the compiler pipeline.

        See also ``RecordLLVMPassTimings``

        Parameters
        ----------
        name: str
            Name for the records.
        """
    def _append(self, name, timings) -> None:
        """Append timing records

        Parameters
        ----------
        name: str
            Name for the records.
        timings: ProcessedPassTimings
            the timing records.
        """
    def get_total_time(self):
        """Computes the sum of the total time across all contained timings.

        Returns
        -------
        res: float or None
            Returns the total number of seconds or None if no timings were
            recorded
        """
    def list_longest_first(self):
        """Returns the timings in descending order of total time duration.

        Returns
        -------
        res: List[ProcessedPassTimings]
        """
    @property
    def is_empty(self):
        """
        """
    def summary(self, topn: int = 5):
        """Return a string representing the summary of the timings.

        Parameters
        ----------
        topn: int; optional, default=5.
            This limits the maximum number of items to show.
            This function will show the ``topn`` most time-consuming passes.

        Returns
        -------
        res: str

        See also ``ProcessedPassTimings.summary()``
        """
    def __getitem__(self, i):
        """Get the i-th timing record.

        Returns
        -------
        res: (name, timings)
            A named tuple with two fields:

            - name: str
            - timings: ProcessedPassTimings
        """
    def __len__(self) -> int:
        """Length of this collection.
        """
    def __str__(self) -> str: ...
