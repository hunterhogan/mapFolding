from _typeshed import Incomplete
from collections.abc import Generator
from numba._version import get_versions as get_versions
from numba.core import errors as errors
from numba.core.registry import cpu_target as cpu_target
from numba.tests.support import captured_stdout as captured_stdout

def _get_commit(): ...

commit: Incomplete
github_url: str

def inspect_function(function, target: Incomplete | None = None):
    '''Return information about the support of a function.

    Returns
    -------
    info : dict
        Defined keys:
        - "numba_type": str or None
            The numba type object of the function if supported.
        - "explained": str
            A textual description of the support.
        - "source_infos": dict
            A dictionary containing the source location of each definition.
    '''
def inspect_module(module, target: Incomplete | None = None, alias: Incomplete | None = None) -> Generator[Incomplete]:
    """Inspect a module object and yielding results from `inspect_function()`
    for each function object in the module.
    """

class _Stat:
    """For gathering simple statistic of (un)supported functions"""
    supported: int
    unsupported: int
    def __init__(self) -> None: ...
    @property
    def total(self): ...
    @property
    def ratio(self): ...
    def describe(self): ...
    def __repr__(self) -> str: ...

def filter_private_module(module_components): ...
def filter_tests_module(module_components): ...

_default_module_filters: Incomplete

def list_modules_in_package(package, module_filters=...) -> Generator[Incomplete, None, Incomplete]:
    """Yield all modules in a given package.

    Recursively walks the package tree.
    """

class Formatter:
    """Base class for formatters.
    """
    _fileobj: Incomplete
    def __init__(self, fileobj) -> None: ...
    def print(self, *args, **kwargs) -> None: ...

class HTMLFormatter(Formatter):
    """Formatter that outputs HTML
    """
    def escape(self, text): ...
    def title(self, text) -> None: ...
    def begin_module_section(self, modname) -> None: ...
    def end_module_section(self) -> None: ...
    def write_supported_item(self, modname, itemname, typename, explained, sources, alias) -> None: ...
    def write_unsupported_item(self, modname, itemname) -> None: ...
    def write_statistic(self, stats) -> None: ...

class ReSTFormatter(Formatter):
    """Formatter that output ReSTructured text format for Sphinx docs.
    """
    def escape(self, text): ...
    def title(self, text) -> None: ...
    def begin_module_section(self, modname) -> None: ...
    def end_module_section(self) -> None: ...
    def write_supported_item(self, modname, itemname, typename, explained, sources, alias) -> None: ...
    def write_unsupported_item(self, modname, itemname) -> None: ...
    def write_statistic(self, stat) -> None: ...

def _format_module_infos(formatter, package_name, mod_sequence, target: Incomplete | None = None) -> None:
    """Format modules.
    """
def write_listings(package_name, filename, output_format) -> None:
    '''Write listing information into a file.

    Parameters
    ----------
    package_name : str
        Name of the package to inspect.
    filename : str
        Output filename. Always overwrite.
    output_format : str
        Support formats are "html" and "rst".
    '''

program_description: Incomplete

def main() -> None: ...
