"""I use this module to derive state initialization from counting source.

(AI generated docstring)

This module extracts and rewrites source counting so initialization stops after the first
counted group, then writes the derived function to a configured module.

Contents
--------
Functions
	makeInitializeState
		Write a state initializer that stops after the first counted fold group.
"""
from __future__ import annotations

from astToolkit import Grab, identifierDotAttribute, NodeChanger, Then
from astToolkit.changeDef import inlineFunctionDef
from astToolkit.containers import IngredientsFunction, IngredientsModule, LedgerOfImports
from mapFolding.kitAST import IfThis, toDisk
from mapFolding.kitAST.dataclasses import findDataclass
from mapFolding.kitAST.theSSOT import default
from typing import TYPE_CHECKING
import ast

if TYPE_CHECKING:
	from mapFolding.theTypes import Default
	from pathlib import PurePath
	from typing import Any

def makeInitializeState(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Write a state initializer that stops after the first counted fold group.

	(AI generated docstring)

	You can use this function to inline source counting and derive a state-initialization function.
	The transformation replaces the selected active-leaf loop condition with a comparison of the
	counting field against zero, renames the function, and writes the resulting module.

	Loop Selection
	--------------
	The visitor searches only the first statement in the inlined function body. Generation
	expects the source counting loop to be reachable within that statement.

	Parameters
	----------
	astModule : ast.Module
		Source module syntax expected by this transformation; generation may modify the supplied
		nodes.
	identifiers : Default | None = None
		Identifier and destination defaults; a falsy value selects the base default mapping.
	**override : Any
		Overrides for source callable, target `identifierCallable`, counting field, module name, and
		writer settings.

	Returns
	-------
	pathFilename : PurePath
		Path to the generated Python module written by the transformation.

	Examples
	--------
	The module `mapFolding.kitAST.mapFolding.makeModules` uses this call [1].

		```python
		makeInitializeState(getModule(identifiers=defaultMapFolding), defaultMapFolding)
		```

	References
	----------
	[1] `mapFolding.kitAST.mapFolding.makeModules`
	"""
	identifiers = identifiers or default
	名CallableSource: identifierDotAttribute = override.get('名CallableSource') or identifiers['function']['counting']
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(名CallableSource, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = override.get('identifierCallable') or identifiers['function'].get('initializeState') or 名CallableSource

	_logicalPathDataclass, _identifierDataclass, identifierDataclassInstance = findDataclass(ingredientsFunction)
	名Counting: identifierDotAttribute = override.get('名Counting') or identifiers['variable']['counting']

	NodeChanger(findThis=IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(identifierDataclassInstance, 'leaf1ndex')
		, doThat=Grab.testAttribute(Grab.andDoAllOf([
			Grab.opsAttribute(Then.replaceWith([ast.Eq()]))
			, Grab.leftAttribute(Grab.attrAttribute(Then.replaceWith(名Counting)))]))
	).visit(ingredientsFunction.astFunctionDef.body[0])

	ingredientsModule = IngredientsModule(ingredientsFunction)

	名Module: str = override.get('名Module') or identifiers['module']['initializeState']

	return toDisk(ingredientsModule, identifiers, override, 名Module)
