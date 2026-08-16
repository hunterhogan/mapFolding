"""Make functions that are complementary to the `count` function and are often called by `doTheNeedful`."""
from __future__ import annotations

from astToolkit import Grab, identifierDotAttribute, NodeChanger, Then
from astToolkit.containers import IngredientsFunction, IngredientsModule, LedgerOfImports
from astToolkit.transformationTools import inlineFunctionDef
from mapFolding.kitAST import IfThis
from mapFolding.kitAST.dataclasses import findDataclass
from mapFolding.kitAST.mapFolding._count import toDisk
from mapFolding.kitAST.theSSOT import default
from typing import TYPE_CHECKING
import ast

if TYPE_CHECKING:
	from mapFolding.theTypes import Default
	from pathlib import PurePath
	from typing import Any

def makeInitializeState(astModule: ast.Module, identifiers: Default | None = None, **keywordArguments: Any) -> PurePath:
	"""Generate initialization module."""
	identifiers = identifiers or default
	名CallableSource: identifierDotAttribute = keywordArguments.get('名CallableSource') or identifiers['function']['counting']
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(名CallableSource, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = keywordArguments.get('identifierCallable') or identifiers['function'].get('initializeState') or 名CallableSource

	_logicalPathDataclass, _identifierDataclass, identifierDataclassInstance = findDataclass(ingredientsFunction)
	名Counting: identifierDotAttribute = keywordArguments.get('名Counting') or identifiers['variable']['counting']

	NodeChanger(findThis=IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(identifierDataclassInstance, 'leaf1ndex')
		, doThat=Grab.testAttribute(Grab.andDoAllOf([
			Grab.opsAttribute(Then.replaceWith([ast.Eq()]))
			, Grab.leftAttribute(Grab.attrAttribute(Then.replaceWith(名Counting)))]))
	).visit(ingredientsFunction.astFunctionDef.body[0])

	ingredientsModule = IngredientsModule(ingredientsFunction)

	名Module: str = keywordArguments.get('名Module') or identifiers['module']['initializeState']

	return toDisk(ingredientsModule, identifiers, keywordArguments, 名Module)
