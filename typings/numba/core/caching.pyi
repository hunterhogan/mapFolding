import abc
from _typeshed import Incomplete
from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from numba.core import compiler as compiler, config as config
from numba.core.base import BaseContext as BaseContext
from numba.core.codegen import CodeLibrary as CodeLibrary
from numba.core.compiler import CompileResult as CompileResult
from numba.core.errors import NumbaWarning as NumbaWarning
from numba.core.serialize import dumps as dumps
from numba.misc.appdirs import AppDirs as AppDirs

def _cache_log(msg, *args) -> None: ...

class _Cache(metaclass=ABCMeta):
    @property
    @abc.abstractmethod
    def cache_path(self):
        """
        The base filesystem path of this cache (for example its root folder).
        """
    @abstractmethod
    def load_overload(self, sig, target_context):
        """
        Load an overload for the given signature using the target context.
        The saved object must be returned if successful, None if not found
        in the cache.
        """
    @abstractmethod
    def save_overload(self, sig, data):
        """
        Save the overload for the given signature.
        """
    @abstractmethod
    def enable(self):
        """
        Enable the cache.
        """
    @abstractmethod
    def disable(self):
        """
        Disable the cache.
        """
    @abstractmethod
    def flush(self):
        """
        Flush the cache.
        """

class NullCache(_Cache):
    @property
    def cache_path(self) -> None: ...
    def load_overload(self, sig, target_context) -> None: ...
    def save_overload(self, sig, cres) -> None: ...
    def enable(self) -> None: ...
    def disable(self) -> None: ...
    def flush(self) -> None: ...

class _CacheLocator(metaclass=ABCMeta):
    """
    A filesystem locator for caching a given function.
    """
    def ensure_cache_path(self) -> None: ...
    @abstractmethod
    def get_cache_path(self):
        """
        Return the directory the function is cached in.
        """
    @abstractmethod
    def get_source_stamp(self):
        """
        Get a timestamp representing the source code's freshness.
        Can return any picklable Python object.
        """
    @abstractmethod
    def get_disambiguator(self):
        """
        Get a string disambiguator for this locator's function.
        It should allow disambiguating different but similarly-named functions.
        """
    @classmethod
    def from_function(cls, py_func, py_file) -> None:
        """
        Create a locator instance for the given function located in the
        given file.
        """
    @classmethod
    def get_suitable_cache_subpath(cls, py_file):
        """Given the Python file path, compute a suitable path inside the
        cache directory.

        This will reduce a file path that is too long, which can be a problem
        on some operating system (i.e. Windows 7).
        """

class _SourceFileBackedLocatorMixin:
    """
    A cache locator mixin for functions which are backed by a well-known
    Python source file.
    """
    def get_source_stamp(self): ...
    def get_disambiguator(self): ...
    @classmethod
    def from_function(cls, py_func, py_file): ...

class _UserProvidedCacheLocator(_SourceFileBackedLocatorMixin, _CacheLocator):
    """
    A locator that always point to the user provided directory in
    `numba.config.CACHE_DIR`
    """
    _py_file: Incomplete
    _lineno: Incomplete
    _cache_path: Incomplete
    def __init__(self, py_func, py_file) -> None: ...
    def get_cache_path(self): ...
    @classmethod
    def from_function(cls, py_func, py_file): ...

class _InTreeCacheLocator(_SourceFileBackedLocatorMixin, _CacheLocator):
    """
    A locator for functions backed by a regular Python module with a
    writable __pycache__ directory.
    """
    _py_file: Incomplete
    _lineno: Incomplete
    _cache_path: Incomplete
    def __init__(self, py_func, py_file) -> None: ...
    def get_cache_path(self): ...

class _UserWideCacheLocator(_SourceFileBackedLocatorMixin, _CacheLocator):
    """
    A locator for functions backed by a regular Python module or a
    frozen executable, cached into a user-wide cache directory.
    """
    _py_file: Incomplete
    _lineno: Incomplete
    _cache_path: Incomplete
    def __init__(self, py_func, py_file) -> None: ...
    def get_cache_path(self): ...
    @classmethod
    def from_function(cls, py_func, py_file): ...

class _IPythonCacheLocator(_CacheLocator):
    """
    A locator for functions entered at the IPython prompt (notebook or other).
    """
    _py_file: Incomplete
    _bytes_source: Incomplete
    def __init__(self, py_func, py_file) -> None: ...
    def get_cache_path(self): ...
    def get_source_stamp(self): ...
    def get_disambiguator(self): ...
    @classmethod
    def from_function(cls, py_func, py_file): ...

class _ZipCacheLocator(_SourceFileBackedLocatorMixin, _CacheLocator):
    """
    A locator for functions backed by Python modules within a zip archive.
    """
    _py_file: Incomplete
    _lineno: Incomplete
    _cache_path: Incomplete
    def __init__(self, py_func, py_file) -> None: ...
    @staticmethod
    def _split_zip_path(py_file): ...
    def get_cache_path(self): ...
    def get_source_stamp(self): ...
    @classmethod
    def from_function(cls, py_func, py_file): ...

class CacheImpl(metaclass=ABCMeta):
    """
    Provides the core machinery for caching.
    - implement how to serialize and deserialize the data in the cache.
    - control the filename of the cache.
    - provide the cache locator
    """
    _locator_classes: Incomplete
    _lineno: Incomplete
    _locator: Incomplete
    _filename_base: Incomplete
    def __init__(self, py_func) -> None: ...
    def get_filename_base(self, fullname, abiflags): ...
    @property
    def filename_base(self): ...
    @property
    def locator(self): ...
    @abstractmethod
    def reduce(self, data):
        """Returns the serialized form the data"""
    @abstractmethod
    def rebuild(self, target_context, reduced_data):
        """Returns the de-serialized form of the *reduced_data*"""
    @abstractmethod
    def check_cachable(self, data):
        """Returns True if the given data is cachable; otherwise, returns False."""

class CompileResultCacheImpl(CacheImpl):
    """
    Implements the logic to cache CompileResult objects.
    """
    def reduce(self, cres):
        """
        Returns a serialized CompileResult
        """
    def rebuild(self, target_context, payload):
        """
        Returns the unserialized CompileResult
        """
    def check_cachable(self, cres):
        """
        Check cachability of the given compile result.
        """

class CodeLibraryCacheImpl(CacheImpl):
    """
    Implements the logic to cache CodeLibrary objects.
    """
    _filename_prefix: Incomplete
    def reduce(self, codelib):
        """
        Returns a serialized CodeLibrary
        """
    def rebuild(self, target_context, payload):
        """
        Returns the unserialized CodeLibrary
        """
    def check_cachable(self, codelib):
        """
        Check cachability of the given CodeLibrary.
        """
    def get_filename_base(self, fullname, abiflags): ...

class IndexDataCacheFile:
    """
    Implements the logic for the index file and data file used by a cache.
    """
    _cache_path: Incomplete
    _index_name: Incomplete
    _index_path: Incomplete
    _data_name_pattern: Incomplete
    _source_stamp: Incomplete
    _version: Incomplete
    def __init__(self, cache_path, filename_base, source_stamp) -> None: ...
    def flush(self) -> None: ...
    def save(self, key, data) -> None:
        """
        Save a new cache entry with *key* and *data*.
        """
    def load(self, key):
        """
        Load a cache entry with *key*.
        """
    def _load_index(self):
        """
        Load the cache index and return it as a dictionary (possibly
        empty if cache is empty or obsolete).
        """
    def _save_index(self, overloads) -> None: ...
    def _load_data(self, name): ...
    def _save_data(self, name, data) -> None: ...
    def _data_name(self, number): ...
    def _data_path(self, name): ...
    def _dump(self, obj): ...
    def _open_for_write(self, filepath) -> Generator[Incomplete]:
        """
        Open *filepath* for writing in a race condition-free way (hopefully).
        uuid4 is used to try and avoid name collisions on a shared filesystem.
        """

class Cache(_Cache):
    '''
    A per-function compilation cache.  The cache saves data in separate
    data files and maintains information in an index file.

    There is one index file per function and Python version
    ("function_name-<lineno>.pyXY.nbi") which contains a mapping of
    signatures and architectures to data files.
    It is prefixed by a versioning key and a timestamp of the Python source
    file containing the function.

    There is one data file ("function_name-<lineno>.pyXY.<number>.nbc")
    per function, function signature, target architecture and Python version.

    Separate index and data files per Python version avoid pickle
    compatibility problems.

    Note:
    This contains the driver logic only.  The core logic is provided
    by a subclass of ``CacheImpl`` specified as *_impl_class* in the subclass.
    '''
    _impl_class: Incomplete
    _name: Incomplete
    _py_func: Incomplete
    _impl: Incomplete
    _cache_path: Incomplete
    _cache_file: Incomplete
    def __init__(self, py_func) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def cache_path(self): ...
    _enabled: bool
    def enable(self) -> None: ...
    def disable(self) -> None: ...
    def flush(self) -> None: ...
    def load_overload(self, sig, target_context):
        """
        Load and recreate the cached object for the given signature,
        using the *target_context*.
        """
    def _load_overload(self, sig, target_context): ...
    def save_overload(self, sig, data) -> None:
        """
        Save the data for the given signature in the cache.
        """
    def _save_overload(self, sig, data) -> None: ...
    def _guard_against_spurious_io_errors(self) -> Generator[None]: ...
    def _index_key(self, sig, codegen):
        """
        Compute index key for the given signature and codegen.
        It includes a description of the OS, target architecture and hashes of
        the bytecode for the function and, if the function has a __closure__,
        a hash of the cell_contents.
        """

class FunctionCache(Cache):
    """
    Implements Cache that saves and loads CompileResult objects.
    """
    _impl_class = CompileResultCacheImpl

_lib_cache_prefixes: Incomplete

def make_library_cache(prefix):
    """
    Create a Cache class for additional compilation features to cache their
    result for reuse.  The cache is saved in filename pattern like
    in ``FunctionCache`` but with additional *prefix* as specified.
    """
