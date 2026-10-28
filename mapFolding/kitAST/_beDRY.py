"""Apply common syntax transformations and write generated modules.

(AI generated docstring)

You can use this module to remove or rename function definitions, rename matching name nodes,
and resolve output settings before writing source containers from `astToolkit`.

Contents
--------
Functions
	removeFunctionDef
		Remove a function definition from `node` when the function name matches `identifier`.
	renameFunctionDef
		Rename a function definition in `node` when the function name matches `identifier`.
	renameName
		Rename matching name nodes in the supplied syntax tree.
	setDatatypeViaImport
		Replace datatype import aliases in generated function and module containers.
	toDisk
		Write a generated module to the resolved output filename.
"""
from __future__ import annotations

from astToolkit import Be, Grab, NodeChanger, Then
from mapFolding.kitAST import IfThis
from mapFolding.kitAST.paths import getPathFilename
from mapFolding.theSSOT import settingsPackage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit.containers import IngredientsFunction, IngredientsModule
	from hunterMakesPy import identifierDotAttribute
	from mapFolding.kitAST import Settings形
	from mapFolding.theTypes import Any, Default
	from os import PathLike
	from pathlib import PurePath
	import ast

def removeFunctionDef(identifier: str, node: ast.AST) -> None:
	"""Remove a function definition from `node` when the function name matches `identifier`.

	(AI generated docstring)

	You can use this function to delete a `FunctionDef` from a parsed AST subtree without writing
	the `NodeChanger` traversal inline. The function mutates `node` in place and returns
	`None`. The function targets each `FunctionDef.name` value that matches `identifier` through
	`IfThis.isIdentifier`.

	Parameters
	----------
	identifier : str
		The function name to match for removal.
	node : ast.AST
		The AST subtree to traverse and mutate.

	See Also
	--------
	`renameFunctionDef`
		Rename a matching `FunctionDef` instead of removing the `FunctionDef`.

	Examples
	--------
	In `mapFolding.kitAST.matrixMeanders.makeModules.makeCountBigInt`, the generated dispatcher
	function is removed with the following call.

		```python
		removeFunctionDef(名CallableDispatcher, astModule)
		```
	"""
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(identifier)), Then.removeIt).visit(node)

def renameFunctionDef(identifier: str, identifierNew: str, node: ast.AST) -> None:
	"""Rename a function definition in `node` when the function name matches `identifier`.

	(AI generated docstring)

	You can use this function to retitle a `FunctionDef` inside a parsed AST subtree without
	writing the `NodeChanger` traversal inline. The function mutates `node` in place and
	returns `None`. The function only rewrites `FunctionDef.name` values that match `identifier`
	through `IfThis.isIdentifier`.

	Parameters
	----------
	identifier : str
		The current function name to match.
	identifierNew : str
		The replacement function name written into `FunctionDef.name`.
	node : ast.AST
		The AST subtree to traverse and mutate.

	See Also
	--------
	`renameName`
		Rename matching `Name` nodes when the rewrite target is not a `FunctionDef`.

	Examples
	--------
	In `mapFolding.kitAST.matrixMeanders.makeModules.makeCountBigInt`, the source counting
	function is renamed before the generated module is written.

		```python
		renameFunctionDef(defaultMatrixMeanders['function']['counting'], 名Callable, astModule)
		```
	"""
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(identifier)), Grab.nameAttribute(Then.replaceWith(identifierNew))).visit(node)

def renameName(identifier: str, identifierNew: str, node: ast.AST) -> None:
	"""Rename matching name nodes in the supplied syntax tree.

	(AI generated docstring)

	You can use this function to change each matching `ast.Name.id` in `node` from
	`identifier` to `identifierNew`. The function updates the supplied tree in place and returns
	`None`. Attribute names and function-definition names are outside this selection.

	Parameters
	----------
	identifier : str
		Current name identifier to match.
	identifierNew : str
		Replacement name identifier.
	node : ast.AST
		Syntax tree or subtree to traverse and modify.
	"""
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(identifier)), Grab.idAttribute(Then.replaceWith(identifierNew))).visit(node)

# TODO Use this concept in general modules, not just custom jobs.
def setDatatypeViaImport(ingredientsFunction: IngredientsFunction, ingredientsModule: IngredientsModule, boxOfSettings形: list[Settings形]) -> tuple[IngredientsFunction, IngredientsModule]:
	"""Customize data types in the given ingredients by adjusting imports.

	In the ecosystem of "Ingredients", "Recipes", "DataBaskets," and "shattered dataclasses," a ton of code is dedicated to
	preserving _abstract_ names for datatypes, such as `形ArrayTotalLeaves1D` and `形TotalFolds`. This function well
	illustrates why I put so much effort into preserving the abstract names. (Normally, Python will _immediately_ replace an alias
	name with the type for which it is a proxy.) Because transformed code, even if it has been through 10 transformations (see,
	for example, `mapFolding.synthesized.mapFoldingSymmetric.asynchronous` or its equivalent), ought to still have the abstract
	names, this function gives you the power to change the datatype from numpy to numba and/or from 8-bits to 16-bits merely by
	changing the import statements. You shouldn't need to change any "business" logic.

	This will not remove potentially conflicting existing imports from other modules.

	Returns
	-------
	datatypesIngredientsObjects : tuple[IngredientsFunction, IngredientsModule]
		A tuple containing the modified `IngredientsFunction` and `IngredientsModule` with updated imports for the specified datatypes.
	"""
	for datatypeConfig in boxOfSettings形:
		ingredientsFunction.imports.removeImportFrom(None, None, datatypeConfig.datatypeIdentifier)
		ingredientsModule.removeImportFrom(None, None, datatypeConfig.datatypeIdentifier)
		ingredientsFunction.imports.addImportFrom_asStr(datatypeConfig.typeModule, datatypeConfig.typeIdentifier, datatypeConfig.type_asname)

	return ingredientsFunction, ingredientsModule

def toDisk(ingredientsModule: IngredientsModule, identifiers: Default, keywords: dict[str, Any], identifierModule: str, **override: Any) -> PurePath:
	"""Write a generated module to the resolved output filename.

	(AI generated docstring)

	You can use this function to resolve an output path for `ingredientsModule` and write the
	assembled source. Truthy values in `override` precede values in `keywords`, which precede
	`identifiers` for the logical path, root, and package. The function returns the writer path.

	Filename Overrides
	------------------
	`fileExtension` and `pathFilename` are read only from `override`, not from `keywords`.
	An explicit truthy `pathFilename` bypasses filename construction. Otherwise,
	`mapFolding.kitAST.paths.getPathFilename` constructs the destination.

	Parameters
	----------
	ingredientsModule : IngredientsModule
		Assembled source container to write.
	identifiers : Default
		Default package, physical root, and generated logical path.
	keywords : dict[str, Any]
		Caller-supplied root, logical path, and package values used before defaults.
	identifierModule : str
		Module filename stem used unless an explicit destination path is supplied.
	**override : Any
		Explicit `logicalPathInfix`, `pathRoot`, `package`, `fileExtension`, and `pathFilename`
		values.

	Returns
	-------
	pathFilename : PurePath
		Destination path returned by the source writer.

	Examples
	--------
	The module `mapFolding.kitAST.mapFolding._count` uses this call.

		```python
		toDisk(ingredientsModule, identifiers, override, 名Module)
		```
	"""
	logicalPathInfix: identifierDotAttribute = override.get('logicalPathInfix') or keywords.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	pathRoot: PathLike[str] = override.get('pathRoot') or keywords.get('pathRoot') or identifiers['filesystem']['pathRoot']
	identifierPackage: str = override.get('package') or keywords.get('package') or identifiers['module']['package']
	fileExtension: str = override.get('fileExtension') or settingsPackage.fileExtension
	pathFilename: PurePath = override.get('pathFilename') or getPathFilename(pathRoot, logicalPathInfix, identifierModule, fileExtension)
	return ingredientsModule.write_astModule(pathFilename, identifierPackage)
