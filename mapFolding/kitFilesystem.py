#ruff: file-ignore[suspicious-pickle-usage] #=Sin= Centralize all pickle usage here.
"""Name, save, cache, and load folding computation data.

(AI generated docstring)

You can use this module to persist counts, albums, integer triangles, and diagonals. This module
also manages temporary storage for Polars computations and retrieves cached text from URLs.

Contents
--------
Storage root
	pathPolarsDataFrame
		Select the root for temporary Polars storage.
Functions
	Temporary Polars storage
		storePolars
			Materialize successive lazy computations within a temporary-storage context.
	Create paths and filenames
		getPathRootJobDEFAULT
			Create the default job directory.
		makeFilenameArrayFoldings
			Build a filename for sampled array-folding data.
		makeFilenameCount
			Join positional and named filename components.
		makeFilenameFolds
			Encode map dimensions in their supplied order.
		makePathFilenameArrayFoldings
			Combine a sample-data directory and filename.
		makePathFilenameCount
			Resolve an output path and create parent directories.
		makePathFilenameFolds
			Resolve an output path named from map dimensions.
	Confirm write capability
		saveTotalFAILearly
			Leave a test value at a new output path before counting.
	Write
		iterableToCSV
			Write rows to CSV with optional appending.
		saveTotal
			Save an integer with fallback output on failure.
		writeAlbum
			Write each folding as one CSV record.
		writeDiagonal
			Write sorted row, diagonal, and value records.
		writeTriangle
			Write sorted row numbers followed by row values.
	Read and write
		getCacheOrURL
			Retrieve cached text or request updated text from a URL.
	Read
		getDataFrameFoldings
			Load bundled folding data for an elimination state.
		readAlbum
			Load all folding records into a tuple.
		readDataFrame
			Load serialized data as a dataframe.
		readDiagonal
			Read one diagonal from a triangle or diagonal file.
		readText
			Read UTF-8 text.
		readTriangle
			Read numbered integer rows.
		readTriangleValue
			Select a stored term by row number and positional column index.
		streamAlbum
			Yield folding records as the file is read.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack, suppress
from csv import writer as csv_writer
from datetime import datetime, timedelta, UTC
from email.utils import format_datetime
from hunterMakesPy import errorL33T, raiseIfNone
from hunterMakesPy.filesystemToolkit import writeStringToHere
from mapFolding.dataBaskets import ShapeArray
from mapFolding.dataStructures import parseCSVtoIntegers, parseDiagonal, parseTriangle
from mapFolding.theSSOT import settingsPackage
from pathlib import Path, PurePosixPath
from platformdirs import user_data_dir
from sys import modules as sysModules, stdout
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from urllib3 import PoolManager
from urllib3.exceptions import HTTPError
import numpy
import os

if TYPE_CHECKING:
	from _csv import Writer
	from collections.abc import Callable, Generator, Iterable, Mapping
	from io import TextIOWrapper
	from mapFolding._e.dataBaskets import StateElimination
	from mapFolding.theTypes import Folding, 形ArrayArcCode, 形Triangle
	from os import PathLike
	from pandas import DataFrame
	from typing import Any, Literal, Protocol
	from urllib3.response import BaseHTTPResponse
	import polars

	class _PolarsMaterializer(Protocol):
		def __call__(self, dataframe: polars.LazyFrame,
			aggregation: polars.Expr | tuple[polars.Expr, ...] | None = None, *, append: bool = False) -> polars.LazyFrame: ...

pathPolarsDataFrame: Path = Path.cwd()
"""Select the root directory for temporary Polars materializations.

(AI generated docstring)

This value captures the working directory when the module is imported. `storePolars` [1] reads
the current value when creating temporary files. Callers can replace this value with an existing
directory before entering the context.

References
----------
[1] `storePolars`

"""

@contextmanager
def storePolars(group_by: str | None = None) -> Generator[_PolarsMaterializer]:
	"""Materialize lazy computations within a temporary-storage context.

	(AI generated docstring)

	You can use this context manager to save successive Polars lazy computations [1] to temporary
	files and receive a callable that returns a lazy scan of the saved result. The context removes its
	temporary files when the context exits.

	Temporary Files
	---------------
	Files are written under `pathPolarsDataFrame` [2] in Parquet format [3]. The root directory must
	already exist. Polars aggregates and sorts each contribution. With `append=True`, Polars merges
	the contribution into the sorted saved result and aggregates matching keys. Each call streams the
	result to one Parquet file and returns a lazy scan.

	A call with `append=False` starts a generation and retains the preceding generation as its source.
	Calls with `append=True` replace only the accumulated result, preserving that source for further
	contributions. Starting another generation removes the older source. Consume returned scans before
	their generation is removed or the context exits. Grouped results are sorted by `group_by`.
	Cleanup also runs if the body of the context raises an error.

	Parameters
	----------
	group_by : str | None = None
		Grouping column. A column name and aggregation expression combines rows. `None` saves each
		supplied frame directly.

	Yields
	------
	materializePolars : _PolarsMaterializer
		Callable accepting a lazy frame and optional aggregation expressions, and returning a lazy
		scan of the saved result. Supply aggregation expressions when `group_by` is set. Use `append=True`
		to merge another contribution into the generation; this requires `group_by` and a saved result.

	Examples
	--------
	`mapFolding.algorithms.matrixMeandersPolars.count` [4] aggregates contributions by arch code.

	```python
	with storePolars(group_by='arcCode') as materializePolars:
		dataframeMeanders = transition(
			dataframeMeanders, state.bitsLocator, state.arcCodeMAXIMUM, state.bitWidth, materializePolars,
			polars.col('meanders').cast(datatypeMeanders).sum().cast(datatypeMeanders)
		)
	```

	The algorithm consumes each resulting frame within the context before cleanup.

	References
	----------
	[1] Polars lazy-frame reference.
		https://docs.pola.rs/api/python/stable/reference/lazyframe/index.html
	[2] `pathPolarsDataFrame`

	[3] Apache Parquet file format.
		https://parquet.apache.org/docs/file-format/
	[4] `mapFolding.algorithms.matrixMeandersPolars.count`

	"""
	#ruff: ignore[import-outside-top-level] #=SIN= Polars is optional.
	import polars

	storagePolars: ExitStack | None = None
	storagePredecessor: ExitStack | None = None
	dataframeStored: polars.LazyFrame | None = None

	def materializePolars(dataframe: polars.LazyFrame,
		aggregation: polars.Expr | tuple[polars.Expr, ...] | None = None, *, append: bool = False) -> polars.LazyFrame:
		nonlocal dataframeStored, storagePolars, storagePredecessor
		if group_by is not None:
			dataframe = dataframe.sort(group_by, maintain_order=True).group_by(group_by, maintain_order=True).agg(aggregation)
		if append:
			key: str = raiseIfNone(group_by)
			dataframe = raiseIfNone(dataframeStored).merge_sorted(dataframe, key=key).group_by(key, maintain_order=True).agg(aggregation)
			del key
		with ExitStack() as storageSuccessor:
			pathPolarsGeneration: Path = Path(storageSuccessor.enter_context(TemporaryDirectory(prefix='polars_', dir=pathPolarsDataFrame)))
			dataframe.sink_parquet(pathPolarsGeneration / 'dataframe.parquet', engine='streaming')
			dataframe = polars.scan_parquet(pathPolarsGeneration / 'dataframe.parquet')
			if group_by is not None:
				dataframe = dataframe.set_sorted(group_by)
			if append:
				raiseIfNone(storagePolars).close()
			else:
				if storagePredecessor is not None:
					storagePredecessor.close()
				storagePredecessor = storagePolars
			storagePolars = storageSuccessor.pop_all()
			dataframeStored = dataframe
		return dataframe

	try:
		yield materializePolars
	finally:
		if storagePolars is not None:
			storagePolars.close()
		if storagePredecessor is not None:
			storagePredecessor.close()

# TODO This is not tenable. What is the shape?
def extendArray(array: 形ArrayArcCode, length: int) -> 形ArrayArcCode:  # ruff: ignore[undocumented-public-function]
	if isinstance(array, numpy.memmap):
		lengthPrevious: int = len(array)
		array.flush()
		array = numpy.memmap(raiseIfNone(array.filename), dtype=array.dtype, mode='r+', offset=array.offset,
			shape=ShapeArray(length=length, indexes=array.shape[1]))
		array[lengthPrevious:length] = 0
	else:
		array = numpy.pad(array, ((0, length - len(array)), (0, 0)))
	return array

#================== Create appropriate paths and filenames =========================================

def getPathRootJobDEFAULT() -> Path:
	"""Create and return the default directory for computation results.

	(AI generated docstring)

	You can use this function to obtain a writable job-directory location using the package name from
	`mapFolding.theSSOT.settingsPackage` [1]. This function creates the selected directory and missing
	parent directories.

	Directory Selection
	-------------------
	When `'google.colab'` is present in the imported-module registry, the directory is
	`'/content/drive/MyDrive'` followed by the package name. Otherwise, `platformdirs.user_data_dir`
	[2] supplies the user-data directory with the author directory disabled. This function does not
	mount a drive or verify that a Colab drive is mounted.

	Returns
	-------
	pathJobDEFAULT : Path
		Default job directory.

	References
	----------
	[1] `mapFolding.theSSOT.settingsPackage`

	[2] platformdirs directory API.
		https://platformdirs.readthedocs.io/en/latest/api.html
	"""
	if 'google.colab' in sysModules:
		pathJobDEFAULT: Path = Path("/content/drive/MyDrive") / settingsPackage.identifierPackage
	else:
		pathJobDEFAULT = Path(user_data_dir(appname=settingsPackage.identifierPackage, appauthor=False, ensure_exists=True))
	pathJobDEFAULT.mkdir(parents=True, exist_ok=True)
	return pathJobDEFAULT

def makeFilenameArrayFoldings(totalDimensions: int, suffix: str = '.pkl') -> str:
	"""Build the sampled-folding filename for `totalDimensions`.

	(AI generated docstring)

	You can use this function to name sampled folding data for a map with length two in every
	dimension. `makePathFilenameArrayFoldings` [1] combines this filename with a sample-data directory.

	Parameters
	----------
	totalDimensions : int
		Dimension count encoded in the filename.
	suffix : str = '.pkl'
		Suffix appended to the filename stem.

	Returns
	-------
	filenameArrayFoldings : str
		`f'arrayFoldings2上{totalDimensions}Dimensional{suffix}'`.

	References
	----------
	[1] `makePathFilenameArrayFoldings`

	"""
	return makeFilenameCount(f'arrayFoldings2上{totalDimensions}Dimensional', suffix=suffix)

def makeFilenameCount(*underscore: str, suffix: str = '.countTotal', **dash: str) -> str:
	"""Build a count filename from positional and named components.

	(AI generated docstring)

	You can use this function to join computation labels into a filename and append `suffix`.
	This function preserves the supplied component order and does not sanitize characters.

	Parameters
	----------
	*underscore : str
		Positional components joined with underscores.
	suffix : str = '.countTotal'
		Suffix appended after all filename components.
	**dash : str
		Named components added after the positional components. Each key and value is joined
		with a dash, and successive components are joined with underscores.

	Returns
	-------
	filename : str
		Joined components followed by `suffix`. With no components, the result is `suffix`.

	Examples
	--------
	The notebook `easyRun/notebookCount.ipynb` names output from the family, index, and algorithm.

		```python
		pathLikeWrite = getPathRootJobDEFAULT() / makeFilenameCount(kind, str(n), flow)
		```
	"""
	stem: str = '_'.join([*underscore, *map('-'.join, dash.items())])
	return stem + suffix

def makeFilenameFolds(mapShape: tuple[int, ...], suffix: str = '.totalFolds') -> str:
	"""Encode the dimensions in `mapShape` in a folding-count filename.

	(AI generated docstring)

	You can use this function to name an output count from the dimension lengths. This function
	preserves the supplied order; this function does not sort or validate dimensions.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Dimensions to encode, in their supplied order.
	suffix : str = '.totalFolds'
		Suffix appended to the dimension-based stem.

	Returns
	-------
	filenameTotalFolds : str
		The letter `'p'` followed by dimension lengths joined with `'x'`, followed by `suffix`.
		An empty `mapShape` produces `'p'` followed by `suffix`.
	"""
	return makeFilenameCount('p' + 'x'.join(map(str, mapShape)), suffix=suffix)

def makePathFilenameArrayFoldings(totalDimensions: int, pathRoot: PathLike[str] = settingsPackage.pathDataSamples, *, suffix: str = '.pkl') -> Path:
	"""Locate sampled-folding data for `totalDimensions` under `pathRoot`.

	(AI generated docstring)

	You can use this function to combine a sample-data directory with the filename produced by
	`makeFilenameArrayFoldings` [1]. This function returns a path without creating directories.

	Parameters
	----------
	totalDimensions : int
		Dimension count encoded in the filename.
	pathRoot : PathLike[str] = settingsPackage.pathDataSamples
		Root directory for sampled-folding data. The default is captured from the shared package
		settings [2] when this module is imported.
	suffix : str = '.pkl'
		Suffix appended to the filename stem.

	Returns
	-------
	pathFilenameArrayFoldings : Path
		Sample-data path under `pathRoot`, preserving a relative root when supplied.

	References
	----------
	[1] `makeFilenameArrayFoldings`

	[2] `mapFolding.theSSOT.settingsPackage`

	"""
	return Path(pathRoot) / makeFilenameArrayFoldings(totalDimensions, suffix=suffix)

def makePathFilenameCount(pathLikeWrite: PathLike[str] | None = None, *underscore: str, suffix: str = ".countTotal", **dash: str) -> Path:
	"""Resolve a count-output path and create missing parent directories.

	(AI generated docstring)

	You can use this function to choose an output filename from `pathLikeWrite` and computation
	labels. Existing directories receive a filename from `makeFilenameCount` [1]; `None` selects
	the default directory from `getPathRootJobDEFAULT` [2].

	Path Interpretation
	-------------------
	Only an existing directory is treated as a directory. A nonexistent directory-like path is
	treated as a filename. An existing absolute file is returned at its existing location.
	This function does not test file-write permissions.

	Parameters
	----------
	pathLikeWrite : PathLike[str] | None = None
		Existing directory, filename, or file path. A relative nondirectory path is placed under
		the default job directory. An absolute file path keeps its location. An existing relative
		directory receives the generated filename and remains relative.
	*underscore : str
		Positional filename components.
	suffix : str = ".countTotal"
		Suffix used by the generated filename.
	**dash : str
		Named filename components joined as key-value pairs.

	Returns
	-------
	pathFilename : Path
		Resolved output path. Missing parent directories are created.

	References
	----------
	[1] `makeFilenameCount`

	[2] `getPathRootJobDEFAULT`

	"""
	filename: str = makeFilenameCount(*underscore, suffix=suffix, **dash)
	if pathLikeWrite is None:
		pathFilename: Path = getPathRootJobDEFAULT() / filename
	else:
		pathLikeWrite = Path(pathLikeWrite)
		if pathLikeWrite.is_dir():
			pathFilename = pathLikeWrite / filename
		elif pathLikeWrite.is_file() and pathLikeWrite.is_absolute():
			pathFilename = pathLikeWrite
		else:
			pathFilename = getPathRootJobDEFAULT() / pathLikeWrite
		pathFilename.parent.mkdir(parents=True, exist_ok=True)
	return pathFilename

def makePathFilenameFolds(mapShape: tuple[int, ...] = (), pathLikeWrite: PathLike[str] | None = None, *, suffix: str = '.totalFolds') -> Path:
	"""Resolve a folding-count output path named from `mapShape`.

	You can use this function to combine the filename from `makeFilenameFolds` [1] with an output
	location. Path interpretation and directory creation follow `makePathFilenameCount` [2].

	Parameters
	----------
	mapShape : tuple[int, ...] = ()
		Dimensions encoded in the generated filename, preserving their order.
	pathLikeWrite : PathLike[str] | None = None
		Output file or existing directory. `None` selects the default job directory.
	suffix : str = '.totalFolds'
		Suffix appended to the generated filename.

	Returns
	-------
	pathFilenameTotalFolds : Path
		Output path after creating missing parent directories. An existing relative directory can
		produce a relative result.

	References
	----------
	[1] `makeFilenameFolds`

	[2] `makePathFilenameCount`

	"""
	filename: str = makeFilenameFolds(mapShape, suffix)
	return makePathFilenameCount(pathLikeWrite, filename.removesuffix(suffix), suffix=suffix)

#================== Confirm the ability to read or write ===========================================

def saveTotalFAILearly[形PathLike: PathLike[str]](pathFilename: 形PathLike) -> 形PathLike:
	"""Test a new output file before beginning a long computation.

	(AI generated docstring)

	You can use this function to verify that `pathFilename` can be written and read before counting.
	This function rejects an existing path, writes a test integer, reads the integer back, and returns
	the original path object. The test file remains in place.

	Test File Lifetime
	------------------
	This function writes through `hunterMakesPy.filesystemToolkit.writeStringToHere` [1].
	A later call to `saveTotal` [2] normally replaces the test integer with the computed count.
	This function does not reserve the path against concurrent writers or remove the test file
	when subsequent computation fails.

	Parameters
	----------
	pathFilename : 形PathLike
		New output file with an existing parent directory.

	Returns
	-------
	pathFilenameValidated : 形PathLike
		The original path object after a successful write and read.

	Raises
	------
	FileExistsError
		If `pathFilename` already exists.
	FileNotFoundError
		If the parent directory is absent, the test file is absent after writing, or the
		integer read back differs from the test integer.

	References
	----------
	[1] hunterMakesPy filesystem toolkit.
		https://context7.com/hunterhogan/huntermakespy
	[2] `saveTotal`

	"""
	if Path(pathFilename).exists():
		message: str = f"`{pathFilename = }` exists: a battle of overwriting might cause tears."
		raise FileExistsError(message)
	if not Path(pathFilename).parent.exists():
		message = f"I received `{pathFilename = }` 0.000139 seconds ago from a function that promised it created the parent directory, but the parent directory does not exist. Fix that now, so your computation doesn't get deleted later. And be compassionate to others."
		raise FileNotFoundError(message)
	countTotal: int = errorL33T
	writeStringToHere(str(countTotal), pathFilename)
	if not Path(pathFilename).exists():
		message = f"I just wrote a test file to `{pathFilename = }`, but it does not exist. Fix that now, so your computation doesn't get deleted later. And continually improve your empathy skills."
		raise FileNotFoundError(message)
	countTotalRead = int(Path(pathFilename).read_text(encoding="utf-8"))
	if countTotalRead != countTotal:
		message = f"I wrote a test file to `{pathFilename = }` with contents of `{str(countTotal) = }`, but I read `{countTotalRead = }` from the file. Python says the values are not equal. Fix that now, so your computation doesn't get corrupted later. And be pro-social."
		raise FileNotFoundError(message)

	return pathFilename

#================== Write =========================================================================

def iterableToCSV(iterable: Iterable[Any], pathFilename: Path, *, append: bool = False) -> Path:
	"""Write the records in `iterable` to `pathFilename`.

	(AI generated docstring)

	You can use this function to save rows as UTF-8 comma-separated records using the default
	Python `csv.writer` dialect [1]. This function creates missing parent directories and returns
	the supplied path after closing the file.

	Parameters
	----------
	iterable : Iterable[Any]
		Iterable of row-like records accepted by the CSV writer.
	pathFilename : Path
		Destination file.
	append : bool = False
		Append records when `True`; otherwise replace existing file contents.

	Returns
	-------
	pathFilenameWritten : Path
		The supplied `pathFilename`.

	Examples
	--------
	`writeAlbum` [2] writes each folding as a CSV record.

		```python
		return iterableToCSV(album, pathFilename)
		```

	References
	----------
	[1] Python `csv.writer`.
		https://docs.python.org/3/library/csv.html#csv.writer
	[2] `writeAlbum`

	"""
	pathFilename.parent.mkdir(parents=True, exist_ok=True)
	with pathFilename.open(encoding="utf-8", mode="a" if append else "w", newline="") as streamWrite:
		csvWriter: Writer = csv_writer(streamWrite)
		csvWriter.writerows(iterable)
	return pathFilename

def saveTotal(pathFilename: PathLike[str], countTotal: int) -> PurePosixPath:
	"""Save `countTotal` with fallback file output and printed recovery output.

	(AI generated docstring)

	You can use this function to persist a completed integer count. This function first writes to
	`pathFilename` using `hunterMakesPy.filesystemToolkit.writeStringToHere` [1]. If that attempt
	fails, this function prints the count and attempts a file in the working directory.

	Fallback Output
	---------------
	The primary attempt catches `BaseException`, including interruption exceptions. The fallback
	filename depends on the remainder of `countTotal` divided by three and is not unique. The
	fallback can overwrite an earlier fallback file. If the fallback fails, this function attempts
	to print the count again. A failure of that final output operation can still propagate.

	Parameters
	----------
	pathFilename : PathLike[str]
		Preferred output file.
	countTotal : int
		Completed count to write as decimal text.

	Returns
	-------
	pathFilenameWritten : PurePosixPath
		Path representation of the successful output location. If the fallback also fails,
		the result is `PurePosixPath('')`, which renders as `'.'`.

	References
	----------
	[1] hunterMakesPy filesystem toolkit.
		https://context7.com/hunterhogan/huntermakespy
	"""
	try:
		pathFilenameWritten: Path | str = writeStringToHere(str(countTotal), pathFilename)
	#ruff: ignore[blind-except] #=Sin= The total must print.
	except BaseException as ERRORmessage:
		#ruff: ignore[too-many-statements-in-try-clause] #=Sin= Is it a sin to ignore a pretentious rule?
		try:
			stdout.write((banner := '\n' + ' '.join(['countTotal'] * 5) + '\n') + f"\n{countTotal = }\n" + banner)
			stdout.writelines(str(ERRORmessage))
			stdout.write(banner + f"\n{countTotal = }\n" + banner)
			#ruff: ignore[os-getcwd, os-path-join] #=Sin= `try` used pathlib, so I'm using builtin here.
			pathFilenameWritten = os.path.join(os.getcwd(), 'countTotal' + ''.join(((countTotal % 3) + 2) * ['YO_']) + '.txt')
			#ruff: ignore[builtin-open, open-file-with-context-handler] #=Sin= `try` used pathlib and context handler.
			streamWriteFallback: TextIOWrapper = open(pathFilenameWritten, 'w', encoding='utf-8')
			streamWriteFallback.write(str(countTotal))
			streamWriteFallback.close()
			stdout.write(pathFilenameWritten)
		#ruff: ignore[bare-except] #=Sin= The total must print.
		except:
			stdout.write(str(countTotal))
			pathFilenameWritten = ''
	return PurePosixPath(pathFilenameWritten)

def writeAlbum(album: Iterable[Folding], pathFilename: Path) -> Path:
	"""Write each folding in `album` as one CSV record.

	(AI generated docstring)

	You can use this function to persist a collection of foldings for `readAlbum` [1] or
	`streamAlbum` [2]. This function delegates writing to `iterableToCSV` [3], which creates missing
	parent directories and replaces existing file contents.

	Parameters
	----------
	album : Iterable[Folding]
		Foldings in their supplied order, with each tuple containing leaf labels by pile position.
	pathFilename : Path
		Destination CSV file.

	Returns
	-------
	pathFilenameWritten : Path
		The supplied path after all records have been written.

	References
	----------
	[1] `readAlbum`

	[2] `streamAlbum`

	[3] `iterableToCSV`

	"""
	return iterableToCSV(album, pathFilename)

def writeTriangle(triangle: 形Triangle, pathFilename: Path) -> Path:
	"""Save numbered integer rows from `triangle` to a CSV file.

	(AI generated docstring)

	You can use this function to persist a triangle for later loading with `readTriangle` [1].
	This function writes the rows to `pathFilename` and returns `pathFilename` after closing the file.

	CSV Layout
	----------
	Each record contains the row number followed by that row's values, without a header. Records
	appear in ascending row-number order. Values within each row retain their supplied order, and
	an empty row contains only the row number. The file uses UTF-8 and the default `csv.writer`
	dialect [2]. `iterableToCSV` [3] creates missing parent directories and replaces existing file
	contents. An empty `triangle` produces an empty file. Row lengths are not validated.

	Parameters
	----------
	triangle : dict[int, tuple[int, ...]]
		Row numbers mapped to ordered integer values.
	pathFilename : Path
		Destination file. Existing contents are overwritten.

	Returns
	-------
	pathFilenameWritten : Path
		The supplied `pathFilename` after writing completes.

	Writing Failures
	----------------
	This function propagates `OSError` if creating parent directories or opening or writing the
	file fails.

	Examples
	--------
	The writer in `easyRun/reduceArches.py` saves accumulated rows using its existing `triangle`
	and destination directory `pathWrite`.

		```python
		from mapFolding.kitFilesystem import writeTriangle

		writeTriangle(triangle, pathWrite / 'A287548ADDENDUM.csv')
		```

	References
	----------
	[1] `readTriangle`

	[2] csv.writer - Python standard library.
		https://docs.python.org/3/library/csv.html#csv.writer
	[3] `iterableToCSV`

	"""
	return iterableToCSV(((rowNumber, *sequence_k) for rowNumber, sequence_k in sorted(triangle.items())), pathFilename)

def writeDiagonal(diagonal: Mapping[int, int], pathFilename: Path, 次diagonal: int, *, append: bool = False) -> Path:
	"""Write `diagonal` as sorted row, diagonal-position, and value records.

	(AI generated docstring)

	You can use this function to save one diagonal in the three-column format accepted by
	`readDiagonal` [1] with `formatData='diagonalCSV'`. Writing uses `iterableToCSV` [2].

	CSV Layout
	----------
	Records contain the row number, `次diagonal`, and value, in ascending row-number order with
	no header. Appending duplicate row numbers for the same diagonal produces a file that the
	diagonal parser rejects.

	Parameters
	----------
	diagonal : Mapping[int, int]
		Row numbers mapped to the stored diagonal values.
	pathFilename : Path
		Destination CSV file.
	次diagonal : int
		Diagonal position repeated in every output record. This function does not validate the value.
	append : bool = False
		Append records when `True`; otherwise replace existing file contents.

	Returns
	-------
	pathFilenameWritten : Path
		The supplied path after writing completes.

	Examples
	--------
	The runner in `easyRun/triangleSemiMake.py` appends a completed count to a diagonal file.

		```python
		writeDiagonal(
			{n: sum(state.lookupMeanders.values()) + state.countAddend},
			makePathFilenameDiagonal(次diagonal), 次diagonal, append=True
		)
		```

	References
	----------
	[1] `readDiagonal`

	[2] `iterableToCSV`

	"""
	return iterableToCSV(((row[0], 次diagonal, row[1]) for row in sorted(diagonal.items())), pathFilename, append=append)

#================== Read and write ================================================================

def getCacheOrURL(pathFilenameCache: Path, cacheDays: int, url: str) -> str:
	"""Read cached text or request updated text from `url`.

	You can use this function to retrieve UTF-8 text while reusing `pathFilenameCache`. A recent
	cache returns immediately. A stale or missing cache triggers an HTTP request with urllib3 [1].
	Suppressed request failures leave the previously read text unchanged.

	Cache Refresh
	-------------
	A stale cache adds the HTTP `If-Modified-Since` header [2]. Status 304 updates the cache
	modification time; status 200 decodes the response and writes the new text using
	`hunterMakesPy.filesystemToolkit.writeStringToHere` [3]. Other statuses retain the earlier text.
	The request disables retries. This function suppresses urllib3 `HTTPError` and `AttributeError`,
	but filesystem and text-decoding errors can propagate.

	Parameters
	----------
	pathFilenameCache : Path
		Local cache file.
	cacheDays : int
		Cache lifetime in days, measured from the file modification time.
	url : str
		Source URL. The protocol prefix is checked only when a request is needed.

	Returns
	-------
	data : str
		Cached or retrieved text. A missing cache and an unsuccessful request produce an empty string.

	Raises
	------
	ValueError
		If a request is needed and `url` does not start with `'http:'` or `'https:'`.

	References
	----------
	[1] urllib3 documentation.
		https://urllib3.readthedocs.io/en/stable/
	[2] RFC 9110, HTTP `If-Modified-Since`.
		https://www.rfc-editor.org/rfc/rfc9110.html#name-if-modified-since
	[3] hunterMakesPy filesystem toolkit.
		https://context7.com/hunterhogan/huntermakespy
	"""
	preferCache: bool = False
	data: str = ''
	cacheDatetime: datetime | None = None

	if pathFilenameCache.exists():
		cacheDatetime = datetime.fromtimestamp(pathFilenameCache.stat().st_mtime, tz=UTC)
		preferCache = datetime.now(tz=UTC) - cacheDatetime < timedelta(days=cacheDays)
		data = pathFilenameCache.read_text(encoding="utf-8")

	if not preferCache:
		if not url.startswith(("http:", "https:")):
			message: str = f"I received {url = }, but it must start with 'http:' or 'https:'"
			raise ValueError(message)

		headers: dict[str, str] | None = None
		if cacheDatetime is not None:
			headers = {"If-Modified-Since": format_datetime(cacheDatetime, usegmt=True)}

		httpPoolManager = PoolManager(retries=False)
		with suppress(HTTPError, AttributeError):
			response: BaseHTTPResponse = httpPoolManager.request("GET", url, headers=headers, preload_content=True, decode_content=True)
			if response.status == 304:  # Not Modified
				pathFilenameCache.touch()  # Update cache file's modification time to local time.
			elif response.status == 200:
				writeStringToHere(data := response.data.decode("utf-8"), pathFilenameCache)
		httpPoolManager.clear()

	return data

#================== Read ==========================================================================

def _csvTo_int(pathFilename: Path) -> Iterable[tuple[int, ...]]:
	with pathFilename.open(encoding="utf-8", mode="r", newline="") as streamRead:
		yield from parseCSVtoIntegers(streamRead)

def getDataFrameFoldings(state: StateElimination) -> DataFrame:
	"""Load bundled array-folding data for the dimension count in `state`.

	(AI generated docstring)

	You can use this function to obtain sample data for an elimination state. This function resolves
	the standard sample-data path with `makePathFilenameArrayFoldings` [1] and loads the file with
	`readDataFrame` [2]. Loading errors propagate to the caller.

	Parameters
	----------
	state : StateElimination
		State providing `state.totalDimensions`.

	Returns
	-------
	dataframeFoldings : DataFrame
		Sample data as a pandas dataframe [3].

	See Also
	--------
	`readDataFrame`
		Load a dataframe from an explicitly chosen file.

	References
	----------
	[1] `makePathFilenameArrayFoldings`

	[2] `readDataFrame`

	[3] pandas `DataFrame`.
		https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
	"""
	return readDataFrame(makePathFilenameArrayFoldings(state.totalDimensions))

def readAlbum(pathFilename: Path) -> tuple[Folding, ...]:
	"""Load all foldings from `pathFilename` into a tuple.

	(AI generated docstring)

	You can use this function to read an album saved by `writeAlbum` [1]. This function converts each
	nonempty CSV record [2] to a tuple of integers and loads every record into memory.

	Parameters
	----------
	pathFilename : Path
		UTF-8 album file without a header.

	Returns
	-------
	album : tuple[Folding, ...]
		Foldings in file order. An empty file returns an empty tuple.

	See Also
	--------
	`streamAlbum`
		Iterate over folding records as the file is read.

	References
	----------
	[1] `writeAlbum`

	[2] Python `csv.reader`.
		https://docs.python.org/3/library/csv.html#csv.reader
	"""
	return tuple(_csvTo_int(pathFilename))

def readDataFrame(pathFilename: PathLike[str]) -> DataFrame:
	"""Load serialized data from `pathFilename` as a dataframe.

	(AI generated docstring)

	You can use this function to load stored folding data into a pandas dataframe [1].
	This function reads the file with `pandas.read_pickle` [2] and passes the loaded object to the
	dataframe constructor. Loading and construction errors propagate to the caller.

	Parameters
	----------
	pathFilename : PathLike[str]
		File containing data serialized for the pickle reader.

	Returns
	-------
	dataframeFoldings : DataFrame
		Dataframe constructed from the loaded object.

	See Also
	--------
	`getDataFrameFoldings`
		Load sample data using the dimension count in an elimination state.

	References
	----------
	[1] pandas `DataFrame`.
		https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
	[2] pandas `read_pickle`.
		https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html
	"""
	#=Sin= `pandas` is optional.
	#ruff: ignore[import-outside-top-level]
	import pandas
	return pandas.DataFrame(pandas.read_pickle(pathFilename))

def readDiagonal(pathFilename: Path, 次diagonal: int, *, formatData: Literal['triangleCSV', 'diagonalCSV'] = 'triangleCSV', rowLength: Callable[[int], int] | None = None, fromRight: bool = True) -> dict[int, int]:
	"""Load diagonal position `次diagonal` from `pathFilename`.

	(AI generated docstring)

	You can use this function to read one diagonal from a triangle file or a three-column diagonal
	file. This function reads UTF-8 text with `readText` [1] and delegates row selection and
	validation to `mapFolding.dataStructures.parseDiagonal` [2].

	Parameters
	----------
	pathFilename : Path
		File containing numbered integer records without a header.
	次diagonal : int
		Diagonal position to select. Triangle selection normally starts at one.
	formatData : {'triangleCSV', 'diagonalCSV'} = 'triangleCSV'
		Triangle records contain a row number followed by row values. Diagonal records contain
		exactly a row number, diagonal position, and value.
	rowLength : Callable[[int], int] | None = None
		Complete row length from a row number for right-edge triangle selection. `None` uses
		stored row lengths. Diagonal records ignore this argument.
	fromRight : bool = True
		Select relative to the right edge of triangle rows when `True`, or the left edge when
		`False`. Diagonal records ignore this argument.

	Returns
	-------
	diagonal : dict[int, int]
		Selected values keyed by ascending row number.

	Validation Failures
	-------------------
	ValueError
		If the format is unsupported, a field is not an integer, triangle selection uses a
		negative position, or diagonal records have the wrong length or duplicate selected rows.

	References
	----------
	[1] `readText`

	[2] `mapFolding.dataStructures.parseDiagonal`

	"""
	return parseDiagonal(readText(pathFilename), 次diagonal, formatData=formatData, rowLength=rowLength, fromRight=fromRight)

def readText(pathFilename: Path) -> str:
	"""Read the contents of `pathFilename` as UTF-8 text.

	(AI generated docstring)

	You can use this function to load a text file into one string. File access and decoding errors
	propagate to the caller.

	Parameters
	----------
	pathFilename : Path
		Text file to read.

	Returns
	-------
	contents : str
		Decoded file contents.

	Examples
	--------
	`mapFolding.oeis.__main__.readBFileTriangle` [1] reads text before parsing stored sequence rows.

		```python
		readText(pathFilename)
		```

	References
	----------
	[1] `mapFolding.oeis.__main__.readBFileTriangle`

	"""
	return pathFilename.read_text(encoding='utf-8')

def readTriangle(pathFilename: Path) -> dict[int, tuple[int, ...]]:
	"""Load numbered integer rows from a triangle CSV file.

	(AI generated docstring)

	You can use this function to load rows previously saved by `writeTriangle` [1]. This function
	reads all of `pathFilename` into memory and returns a dictionary mapping row numbers to tuples
	of values.

	CSV Layout
	----------
	The file must contain UTF-8 text with no header. Each CSV record [2] begins with an integer row
	number followed by integer values. `mapFolding.dataStructures.parseTriangle` [3] ignores empty
	records and preserves the supplied row order. A record containing only a row number produces an
	empty tuple. Repeated row numbers keep the last values at the first occurrence's dictionary
	position [4]. An empty file produces an empty dictionary. Row lengths and consecutive numbering
	are not validated.

	Parameters
	----------
	pathFilename : Path
		Triangle file to read.

	Returns
	-------
	triangle : dict[int, tuple[int, ...]]
		Row numbers mapped to all values after each row number.

	Reading Failures
	----------------
	This function propagates `OSError` if opening or reading `pathFilename` fails, and
	`UnicodeDecodeError` if the file does not contain valid UTF-8 text. A field that cannot be
	converted to an integer, including a header or empty field, propagates `ValueError` from
	the parser.

	References
	----------
	[1] `writeTriangle`

	[2] csv.reader - Python standard library.
		https://docs.python.org/3/library/csv.html#csv.reader
	[3] `mapFolding.dataStructures.parseTriangle`

	[4] Dictionary insertion order - Python standard library.
		https://docs.python.org/3/library/stdtypes.html#mapping-types-dict
	"""
	return parseTriangle(readText(pathFilename))

def readTriangleValue(pathFilename: Path, rowNumber: int, columnNumber: int) -> int:
	"""Read one term by `rowNumber` and positional `columnNumber`.

	(AI generated docstring)

	You can use this function to select a value from a stored integer triangle. This function loads
	the entire file with `readTriangle` [1], then indexes the requested row tuple.

	Usage
	-----
	No application invocation was found in the codebase.

	Parameters
	----------
	pathFilename : Path
		Triangle CSV file to load.
	rowNumber : int
		Stored row label, rather than a positional row index.
	columnNumber : int
		Zero-based tuple index within the row. Negative indices select from the row end.

	Returns
	-------
	term : int
		Value at the requested row label and tuple position.

	Validation Failures
	-------------------
	KeyError
		If the stored triangle has no row labeled `rowNumber`.
	IndexError
		If `columnNumber` is outside the selected row tuple.

	References
	----------
	[1] `readTriangle`

	"""
	return readTriangle(pathFilename)[rowNumber][columnNumber]

def streamAlbum(pathFilename: Path) -> Iterable[Folding]:
	"""Yield foldings from `pathFilename` as the records are read.

	(AI generated docstring)

	You can use this function to process an album saved by `writeAlbum` [1] without loading all
	records into memory. Each nonempty CSV record [2] becomes a tuple of integers.

	Open File Lifetime
	------------------
	The file is opened when iteration starts and remains open until the generator finishes or is
	closed. Retain and close the generator if iteration stops before exhausting the file.

	Parameters
	----------
	pathFilename : Path
		UTF-8 album file without a header.

	Yields
	------
	folding : Folding
		One folding in file order.

	See Also
	--------
	`readAlbum`
		Load all folding records into a tuple.

	References
	----------
	[1] `writeAlbum`

	[2] Python `csv.reader`.
		https://docs.python.org/3/library/csv.html#csv.reader
	"""
	yield from _csvTo_int(pathFilename)
