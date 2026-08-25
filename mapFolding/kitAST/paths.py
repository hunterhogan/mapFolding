"""Resolve logical module names and output paths for AST-generated modules.

(AI generated docstring)

You can use this module to assemble dotted module paths, load source modules from those paths,
and derive filesystem destinations for generated modules. The module centralizes fallback rules
from `mapFolding.kitAST.theSSOT.default` [1] and `mapFolding.theSSOT.settingsPackage` [2] so
the code-generation subpackage can resolve imports and output filenames consistently.

Contents
--------
Functions
	getLogicalPath
		Assemble a dotted logical module path from package and module components.
	getModule
		Parse a resolved logical module path into an `ast.Module`.
	getPathFilename
		Construct a module filename from logical path components.

References
----------
[1] `mapFolding.kitAST.theSSOT.default`

[2] `mapFolding.theSSOT.settingsPackage`
"""
from __future__ import annotations

from astToolkit.filesystem import parseLogicalPath2astModule
from humpy_cytoolz import get_in
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST.theSSOT import default
from mapFolding.theSSOT import settingsPackage
from pathlib import PurePath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from mapFolding.theTypes import Default
	from os import PathLike
	import ast

def getLogicalPath(identifierPackage: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, *identifierModule: str) -> identifierDotAttribute:
	"""Assemble a dotted logical module path from package and module components.

	You can use this function to join `identifierPackage`, `logicalPathInfix`, and each
	`identifierModule` entry into one dotted logical path. The function returns the resulting
	identifier string after dropping each falsy component, which lets callers omit optional path
	segments without creating repeated separators.

	Parameters
	----------
	identifierPackage : str | None = None
		Package identifier placed at the beginning of the logical path.
	logicalPathInfix : identifierDotAttribute | None = None
		Dot-separated middle segment inserted between `identifierPackage` and `identifierModule`.
	*identifierModule : str
		Module-name segment or segment sequence appended to the logical path.

	Returns
	-------
	logicalPath : identifierDotAttribute
		Dotted logical path assembled from the provided components.

	Examples
	--------
	In `mapFolding.kitAST.mapFolding._count.makeTheorem2`, the import path for the generated
	initializer module is assembled with the following call [1].

		```python
		dotModule: identifierDotAttribute = getLogicalPath(
			名Package,
			logicalPathInfix,
			名ModuleInitializeState,
		)
		```

	References
	----------
	[1] `mapFolding.kitAST.mapFolding._count.makeTheorem2`
	"""
	return '.'.join(filter(None, [identifierPackage, logicalPathInfix, *identifierModule]))

def getModule(
	identifierModule: str | None = None
	, logicalPathInfix: identifierDotAttribute | None = None
	, identifierPackage: str | None = None
	, identifiers: Default | None = None
) -> ast.Module:
	"""Parse a resolved logical module path into an `ast.Module`.

	You can use this function to load a source module AST from explicit identifiers or from the
	nested `identifiers` mapping when explicit values are omitted. The function resolves
	`identifierPackage`, `logicalPathInfix`, and `identifierModule`, then parses the resulting
	logical path into an `ast.Module`.

	Parameters
	----------
	identifierModule : str | None = None
		Module identifier placed at the end of the logical path.
	logicalPathInfix : identifierDotAttribute | None = None
		Dot-separated middle segment between `identifierPackage` and `identifierModule`.
	identifierPackage : str | None = None
		Package identifier placed at the beginning of the logical path.
	identifiers : Default | None = None
		Nested default mapping used when explicit arguments are omitted.

	Returns
	-------
	astModule : ast.Module
		Parsed module AST for the resolved logical path.

	See Also
	--------
	`getLogicalPath`
		Assemble the dotted logical path that this function parses.
	`getPathFilename`
		Build a filesystem path when a generated module needs a destination filename.

	Fallback order
	--------------
	component resolution : behavior
		Each logical-path component is resolved from the explicit argument first, then from
		`identifiers`, then from `mapFolding.kitAST.theSSOT.default` [1]. `identifierModule` must
		resolve to a non-`None` value before parsing begins.

	Examples
	--------
	In `mapFolding.kitAST.mapFolding.makeModules.makeModulesMapFolding`, the default map-folding
	algorithm module is loaded with the following call [2].

		```python
		make_jit_module(getModule(identifiers=defaultMapFolding), defaultMapFolding)
		```

	References
	----------
	[1] `mapFolding.kitAST.theSSOT.default`

	[2] `mapFolding.kitAST.mapFolding.makeModules.makeModulesMapFolding`
	"""
	identifierPackage = identifierPackage or get_in(('module', 'package'), identifiers or {}) or default['module']['package']
	logicalPathInfix = logicalPathInfix or get_in(('logicalPath', 'default'), identifiers or {}) or default['logicalPath']['default']
	identifierModule = identifierModule or get_in(('module', 'default'), identifiers or {}) or default['module']['default']
	return parseLogicalPath2astModule(getLogicalPath(identifierPackage, logicalPathInfix, raiseIfNone(identifierModule)))

def getPathFilename(
	pathRoot: PathLike[str] | None = settingsPackage.pathPackage
	, logicalPathInfix: identifierDotAttribute | None = None
	, identifierModule: str = ''
	, fileExtension: str = settingsPackage.fileExtension
) -> PurePath:
	"""Construct a module filename from logical path components.

	You can use this function to translate `pathRoot`, `logicalPathInfix`, `identifierModule`, and
	`fileExtension` into a `PurePath` for a generated Python module. The function appends
	`fileExtension` to `identifierModule`, expands `logicalPathInfix` into directory segments, and
	prepends `pathRoot` when `pathRoot` is not `None`.

	Parameters
	----------
	pathRoot : PathLike[str] | None = settingsPackage.pathPackage
		Base directory for the package structure.
	logicalPathInfix : identifierDotAttribute | None = None
		Dot-separated logical path inserted between `pathRoot` and `identifierModule`.
	identifierModule : str = ''
		Module identifier used as the filename stem.
	fileExtension : str = settingsPackage.fileExtension
		File extension appended to `identifierModule`.

	Returns
	-------
	pathFilename : PurePath
		Complete filesystem path for the generated module file.

	See Also
	--------
	`getLogicalPath`
		Assemble the dotted logical path that often supplies `logicalPathInfix`.
	`getModule`
		Parse a logical module path when a caller needs the module contents instead of a filename.

	Path assembly
	-------------
	directory expansion : behavior
		When `logicalPathInfix` is not `None`, the function splits `logicalPathInfix` on `'.'` and
		inserts each segment as one directory in the returned `PurePath`.

	Examples
	--------
	In `mapFolding.kitAST.numba.kitNumba.make_jit_module`, the destination filename for a generated
	Numba module is derived with the following call [1].

		```python
		pathFilename: PurePath = getPathFilename(pathRoot, logicalPathInfix, identifierModule)
		```

	References
	----------
	[1] `mapFolding.kitAST.numba.kitNumba.make_jit_module`

	"""
	pathFilename = PurePath(identifierModule + fileExtension)
	if logicalPathInfix:
		pathFilename = PurePath(*(str(logicalPathInfix).split('.')), pathFilename)
	if pathRoot:
		pathFilename = PurePath(pathRoot, pathFilename)
	return pathFilename
