"""Rewrite selected identifier-bearing nodes in a Python AST.

(AI generated docstring)

You can use this module to apply a small set of recurring AST rewrites inside the map-folding
code-generation modules. The module wraps `astToolkit` [1] `NodeChanger` patterns that remove a
`FunctionDef`, rename a `FunctionDef`, or rename a `Name` node when the node identifier matches the
requested predicate from `mapFolding.kitAST.IfThis` [2].

Contents
--------
Functions
	removeFunctionDef
		Remove a function definition from an AST when the function name matches `identifier`.
	renameFunctionDef
		Rename a function definition in an AST when the function name matches `identifier`.
	renameName
		Rename `Name` nodes in an AST when the identifier matches `identifier`.

References
----------
[1] astToolkit - Context7
	https://context7.com/hunterhogan/asttoolkit
[2] `mapFolding.kitAST.IfThis`
"""
from __future__ import annotations

from astToolkit import Be, Grab, NodeChanger, Then
from mapFolding.kitAST import IfThis
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import ast

def removeFunctionDef(identifier: str, node: ast.AST) -> None:
	"""Remove a function definition from `node` when the function name matches `identifier`.

	(AI generated docstring)

	You can use this function to delete a `FunctionDef` from a parsed AST subtree without writing
	the `NodeChanger` traversal inline [1]. The function mutates `node` in place and returns
	`None`. The function targets each `FunctionDef.name` value that matches `identifier` through
	`IfThis.isIdentifier` [2].

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
	function is removed with the following call [3].

		```python
		removeFunctionDef(名CallableDispatcher, astModule)
		```

	References
	----------
	[1] astToolkit - Context7
		https://context7.com/hunterhogan/asttoolkit
	[2] `mapFolding.kitAST.IfThis`

	[3] `mapFolding.kitAST.matrixMeanders.makeModules.makeCountBigInt`
	"""
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(identifier)), Then.removeIt).visit(node)

def renameFunctionDef(identifier: str, identifierNew: str, node: ast.AST) -> None:
	"""Rename a function definition in `node` when the function name matches `identifier`.

	(AI generated docstring)

	You can use this function to retitle a `FunctionDef` inside a parsed AST subtree without
	writing the `NodeChanger` traversal inline [1]. The function mutates `node` in place and
	returns `None`. The function only rewrites `FunctionDef.name` values that match `identifier`
	through `IfThis.isIdentifier` [2]. The function does not rewrite call sites or other `Name`
	nodes.

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
	function is renamed before the generated module is written [3].

		```python
		renameFunctionDef(defaultMatrixMeanders['function']['counting'], 名Callable, astModule)
		```

	References
	----------
	[1] astToolkit - Context7
		https://context7.com/hunterhogan/asttoolkit
	[2] `mapFolding.kitAST.IfThis`

	[3] `mapFolding.kitAST.matrixMeanders.makeModules.makeCountBigInt`
	"""
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(identifier)), Grab.nameAttribute(Then.replaceWith(identifierNew))).visit(node)

def renameName(identifier: str, identifierNew: str, node: ast.AST) -> None:
	"""Rename `Name` nodes in `node` when the identifier matches `identifier`.

	(AI generated docstring)

	You can use this function to rewrite `ast.Name` identifiers inside a parsed AST subtree without
	writing the `NodeChanger` traversal inline [1]. The function mutates `node` in place and
	returns `None`. The function only rewrites `Name.id` values that match `identifier` through
	`IfThis.isIdentifier` [2]. The function does not rewrite `Attribute.attr` values or
	`FunctionDef.name` values.

	Parameters
	----------
	identifier : str
		The current identifier to match in `Name.id`.
	identifierNew : str
		The replacement identifier written into `Name.id`.
	node : ast.AST
		The AST subtree to traverse and mutate.

	See Also
	--------
	`renameFunctionDef`
		Rename matching `FunctionDef.name` values when the rewrite target is a function definition.

	Examples
	--------
	In `mapFolding.kitAST.matrixMeanders.makeModules.makeShare`, the generated callable rewrites
	`int` names to `形ArcCode` before the module is written [3].

		```python
		renameName('int', '形ArcCode', ingredients.astFunctionDef)
		```

	References
	----------
	[1] astToolkit - Context7
		https://context7.com/hunterhogan/asttoolkit
	[2] `mapFolding.kitAST.IfThis`

	[3] `mapFolding.kitAST.matrixMeanders.makeModules.makeShare`
	"""
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(identifier)), Grab.idAttribute(Then.replaceWith(identifierNew))).visit(node)
