import pyflakes.messages
import pyflakes.reporter
from _typeshed import Incomplete
from typing import Any, Callable, IO, Iterable, Mapping, MutableMapping, Sequence

__version__: str
_LOGGER: Incomplete
ATOMS: Incomplete
EXCEPT_REGEX: Incomplete
PYTHON_SHEBANG_REGEX: Incomplete
MAX_PYTHON_FILE_DETECTION_BYTES: int
IGNORE_COMMENT_REGEX: Incomplete

def standard_paths() -> Iterable[str]:
    """Yield paths to standard modules."""
def standard_package_names() -> Iterable[str]:
    """Yield standard module names."""

IMPORTS_WITH_SIDE_EFFECTS: Incomplete
BINARY_IMPORTS: Incomplete
SAFE_IMPORTS: Incomplete

def unused_import_line_numbers(messages: Iterable[pyflakes.messages.Message]) -> Iterable[int]:
    """Yield line numbers of unused imports."""
def unused_import_module_name(messages: Iterable[pyflakes.messages.Message]) -> Iterable[tuple[int, str]]:
    """Yield line number and module name of unused imports."""
def star_import_used_line_numbers(messages: Iterable[pyflakes.messages.Message]) -> Iterable[int]:
    """Yield line number of star import usage."""
def star_import_usage_undefined_name(messages: Iterable[pyflakes.messages.Message]) -> Iterable[tuple[int, str, str]]:
    """Yield line number, undefined name, and its possible origin module."""
def unused_variable_line_numbers(messages: Iterable[pyflakes.messages.Message]) -> Iterable[int]:
    """Yield line numbers of unused variables."""
def duplicate_key_line_numbers(messages: Iterable[pyflakes.messages.Message], source: str) -> Iterable[int]:
    """Yield line numbers of duplicate keys."""
def create_key_to_messages_dict(messages: Iterable[pyflakes.messages.MultiValueRepeatedKeyLiteral]) -> Mapping[Any, Iterable[pyflakes.messages.MultiValueRepeatedKeyLiteral]]:
    """Return dict mapping the key to list of messages."""
def check(source: str) -> Iterable[pyflakes.messages.Message]:
    """Return messages from pyflakes."""

class StubFile:
    """Stub out file for pyflakes."""
    def write(self, *_: Any) -> None:
        """Stub out."""

class ListReporter(pyflakes.reporter.Reporter):
    """Accumulate messages in messages list."""
    messages: list[pyflakes.messages.Message]
    def __init__(self) -> None:
        """Initialize.

        Ignore errors from Reporter.
        """
    def flake(self, message: pyflakes.messages.Message) -> None:
        """Accumulate messages."""

def extract_package_name(line: str) -> str | None:
    """Return package name in import statement."""
def multiline_import(line: str, previous_line: str = '') -> bool:
    """Return True if import is spans multiples lines."""
def multiline_statement(line: str, previous_line: str = '') -> bool:
    """Return True if this is part of a multiline statement."""

class PendingFix:
    """Allows a rewrite operation to span multiple lines.

    In the main rewrite loop, every time a helper function returns a
    ``PendingFix`` object instead of a string, this object will be called
    with the following line.
    """
    accumulator: Incomplete
    def __init__(self, line: str) -> None:
        """Analyse and store the first line."""
    def __call__(self, line: str) -> PendingFix | str:
        """Process line considering the accumulator.

        Return self to keep processing the following lines or a string
        with the final result of all the lines processed at once.
        """

def _valid_char_in_line(char: str, line: str) -> bool:
    """Return True if a char appears in the line and is not commented."""
def _top_module(module_name: str) -> str:
    """Return the name of the top level module in the hierarchy."""
def _modules_to_remove(unused_modules: Iterable[str], safe_to_remove: Iterable[str] = ...) -> Iterable[str]:
    """Discard unused modules that are not safe to remove from the list."""
def _segment_module(segment: str) -> str:
    """Extract the module identifier inside the segment.

    It might be the case the segment does not have a module (e.g. is composed
    just by a parenthesis or line continuation and whitespace). In this
    scenario we just keep the segment... These characters are not valid in
    identifiers, so they will never be contained in the list of unused modules
    anyway.
    """

class FilterMultilineImport(PendingFix):
    '''Remove unused imports from multiline import statements.

    This class handles both the cases: "from imports" and "direct imports".

    Some limitations exist (e.g. imports with comments, lines joined by ``;``,
    etc). In these cases, the statement is left unchanged to avoid problems.
    '''
    IMPORT_RE: Incomplete
    INDENTATION_RE: Incomplete
    BASE_RE: Incomplete
    SEGMENT_RE: Incomplete
    IDENTIFIER_RE: Incomplete
    remove: Iterable[str]
    parenthesized: bool
    base: Incomplete
    give_up: bool
    def __init__(self, line: str, unused_module: Iterable[str] = (), remove_all_unused_imports: bool = False, safe_to_remove: Iterable[str] = ..., previous_line: str = '') -> None:
        """Receive the same parameters as ``filter_unused_import``."""
    def is_over(self, line: str | None = None) -> bool:
        """Return True if the multiline import statement is over."""
    def analyze(self, line: str) -> None:
        """Decide if the statement will be fixed or left unchanged."""
    def fix(self, accumulated: Iterable[str]) -> str:
        """Given a collection of accumulated lines, fix the entire import."""
    def __call__(self, line: str | None = None) -> PendingFix | str:
        """Accumulate all the lines in the import and then trigger the fix."""

def _filter_imports(imports: Iterable[str], parent: str | None = None, unused_module: Iterable[str] = ()) -> Sequence[str]: ...
def filter_from_import(line: str, unused_module: Iterable[str]) -> str:
    """Parse and filter ``from something import a, b, c``.

    Return line without unused import modules, or `pass` if all of the
    module in import is unused.
    """
def break_up_import(line: str) -> str:
    """Return line with imports on separate lines."""
def filter_code(source: str, additional_imports: Iterable[str] | None = None, expand_star_imports: bool = False, remove_all_unused_imports: bool = False, remove_duplicate_keys: bool = False, remove_unused_variables: bool = False, remove_rhs_for_unused_variables: bool = False, ignore_init_module_imports: bool = False) -> Iterable[str]:
    """Yield code with unused imports removed."""
def get_messages_by_line(messages: Iterable[pyflakes.messages.Message]) -> Mapping[int, pyflakes.messages.Message]:
    """Return dictionary that maps line number to message."""
def filter_star_import(line: str, marked_star_import_undefined_name: Iterable[str]) -> str:
    """Return line with the star import expanded."""
def filter_unused_import(line: str, unused_module: Iterable[str], remove_all_unused_imports: bool, imports: Iterable[str], previous_line: str = '') -> PendingFix | str:
    """Return line if used, otherwise return None."""
def filter_unused_variable(line: str, previous_line: str = '', drop_rhs: bool = False) -> str:
    """Return line if used, otherwise return None."""
def filter_duplicate_key(line: str, message: pyflakes.messages.Message, line_number: int, marked_line_numbers: Iterable[int], source: str, previous_line: str = '') -> str:
    """Return '' if first occurrence of the key otherwise return `line`."""
def dict_entry_has_key(line: str, key: Any) -> bool:
    """Return True if `line` is a dict entry that uses `key`.

    Return False for multiline cases where the line should not be removed by
    itself.

    """
def is_literal_or_name(value: str) -> bool:
    """Return True if value is a literal or a name."""
def useless_pass_line_numbers(source: str, ignore_pass_after_docstring: bool = False) -> Iterable[int]:
    '''Yield line numbers of unneeded "pass" statements.'''
def filter_useless_pass(source: str, ignore_pass_statements: bool = False, ignore_pass_after_docstring: bool = False) -> Iterable[str]:
    '''Yield code with useless "pass" lines removed.'''
def get_indentation(line: str) -> str:
    """Return leading whitespace."""
def get_line_ending(line: str) -> str:
    """Return line ending."""
def fix_code(source: str, additional_imports: Iterable[str] | None = None, expand_star_imports: bool = False, remove_all_unused_imports: bool = False, remove_duplicate_keys: bool = False, remove_unused_variables: bool = False, remove_rhs_for_unused_variables: bool = False, ignore_init_module_imports: bool = False, ignore_pass_statements: bool = False, ignore_pass_after_docstring: bool = False) -> str:
    """Return code with all filtering run on it."""
def fix_file(filename: str, args: Mapping[str, Any], standard_out: IO[str] | None = None) -> int:
    """Run fix_code() on a file."""
def _fix_file(input_file: IO[str], filename: str, args: Mapping[str, Any], write_to_stdout: bool, standard_out: IO[str], encoding: str | None = None) -> int: ...
def open_with_encoding(filename: str, encoding: str | None, mode: str = 'r', limit_byte_check: int = -1) -> IO[str]:
    """Return opened file with a specific encoding."""
def detect_encoding(filename: str, limit_byte_check: int = -1) -> str:
    """Return file encoding."""
def _detect_encoding(readline: Callable[[], bytes]) -> str:
    """Return file encoding."""
def get_diff_text(old: Sequence[str], new: Sequence[str], filename: str) -> str:
    """Return text of unified diff between old and new."""
def _split_comma_separated(string: str) -> set[str]:
    """Return a set of strings."""
def is_python_file(filename: str) -> bool:
    """Return True if filename is Python file."""
def is_exclude_file(filename: str, exclude: Iterable[str]) -> bool:
    """Return True if file matches exclude pattern."""
def match_file(filename: str, exclude: Iterable[str]) -> bool:
    """Return True if file is okay for modifying/recursing."""
def find_files(filenames: list[str], recursive: bool, exclude: Iterable[str]) -> Iterable[str]:
    """Yield filenames."""
def process_pyproject_toml(toml_file_path: str) -> MutableMapping[str, Any] | None:
    """Extract config mapping from pyproject.toml file."""
def process_config_file(config_file_path: str) -> MutableMapping[str, Any] | None:
    """Extract config mapping from config file."""
def find_and_process_config(args: Mapping[str, Any]) -> MutableMapping[str, Any] | None: ...
def merge_configuration_file(flag_args: MutableMapping[str, Any]) -> tuple[MutableMapping[str, Any], bool]:
    """Merge configuration from a file into args."""
def _main(argv: Sequence[str], standard_out: IO[str] | None, standard_error: IO[str] | None, standard_input: IO[str] | None = None) -> int:
    """Return exit status.

    0 means no error.
    """
def main() -> int:
    """Command-line entry point."""
