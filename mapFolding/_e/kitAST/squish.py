from __future__ import annotations

from astToolkit import Be, Make, NodeChanger, NodeTourist, parseLogicalPath2astModule, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsModule, LedgerOfImports
from itertools import repeat
from mapFolding import packageSettings
from mapFolding._e.kitAST.infoBooth import default
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterMakesPy import identifierDotAttribute
	from typing import TypeIs
	import ast

pathFilename = Path(*default['logicalPath']['synthetic'].split('.'), 'module1' + packageSettings.fileExtension)

ledgerTYPE_CHECKING = LedgerOfImports()
module1 = IngredientsModule()

def _dissectModule(logicalPath: identifierDotAttribute) -> ast.Module:
	moduleDissect: ast.Module = parseLogicalPath2astModule(logicalPath, optimize=2)

	findThis: Callable[[ast.AST], TypeIs[ast.If]] = Be.If.testIs(Be.Name.idIs('TYPE_CHECKING'.__eq__))
	NodeTourist(findThis, ledgerTYPE_CHECKING.walkThis).visit(moduleDissect)
	NodeChanger(findThis, Then.removeIt).visit(moduleDissect)

	return moduleDissect

def assimilateFunction(logicalPath: identifierDotAttribute, identifierFunction: str) -> None:
	module1.appendIngredientsFunction(astModuleToIngredientsFunction(_dissectModule(logicalPath), identifierFunction))

def assimilateModule(logicalPath: identifierDotAttribute) -> None:
	moduleDissect: ast.Module = _dissectModule(logicalPath)

	module1.imports.walkThis(moduleDissect)
	NodeChanger(Be.Import, Then.removeIt).visit(moduleDissect)
	NodeChanger(Be.ImportFrom, Then.removeIt).visit(moduleDissect)

	NodeChanger(Be.Expr.valueIs(Be.Constant.valueIs(lambda fu: isinstance(fu, str))), Then.removeIt).visit(moduleDissect)

	module1.appendEpilogue(moduleDissect)

listModules: list[identifierDotAttribute] = [
	*tuple(map("{0}._e.{1}".format, repeat(packageSettings.identifierPackage), (
		'theTypes'
		, 'semiotics'
		, 'leafDomains'
		, 'pileOptions'
		, '_disaggregation'
		, '_beDRY'
		, 'dataBaskets'
		, 'filters'
		, 'pinIt'
)))
	, *tuple(map("{0}._e._2上nDimensional.{1}".format, repeat(packageSettings.identifierPackage), (
		'pinIt'
		, 'pinByCrease'
		, 'pinByDomain'
		, 'pinItAnnex'
		, 'semiotics'
		, 'beDRY'
		, 'measure'
		, 'creases'
		, 'leafDomains'
		, 'conditionalOrdering'
		, 'pileOptions'
		, 'filters'
)))
	, f"{default['logicalPath']['algorithm']}.iff"
	, f"{default['logicalPath']['algorithm']}.{default['module']['algorithm']}"
]

listPackages: list[identifierDotAttribute] = [
	*tuple(map("{0}.{1}".format, repeat(packageSettings.identifierPackage), (
		'beDRY'
		, '_e'
		, '_e._2上nDimensional'
)))
]

assimilateFunction('mapFolding.beDRY', 'getLeavesTotal')
assimilateFunction('mapFolding.beDRY', 'defineProcessorLimit')

tuple(map(assimilateModule, listModules))
tuple(map(ledgerTYPE_CHECKING.removeImportFrom, listModules, repeat(None)))
tuple(map(ledgerTYPE_CHECKING.removeImportFrom, listPackages, repeat(None)))
tuple(map(module1.removeImportFrom, listModules, repeat(None)))
tuple(map(module1.removeImportFrom, listPackages, repeat(None)))

ast_stmtTYPE_CHECKING = Make.If(Make.Name('TYPE_CHECKING'), ledgerTYPE_CHECKING.makeList_ast())
module1.appendPrologue(statement=ast_stmtTYPE_CHECKING)

module1.write_astModule(pathFilename, packageSettings.identifierPackage)

"""
# ruff:file-ignore[commented-out-code, print]
if __name__ == "__main__":
	CPUlimit: int | float | None = None
	state: EliminationState = EliminationState((2,) * 4)
	# state = pinPile零Ante首零(state)
	state = pinPilesAtEnds(state, 4)
	state = pinLeavesDimension首二(state)
	# state = pin3beans2(state)
	# state = pin首beans(state)
	# state = pinLeavesDimension一(state)
	# state = pinLeavesDimension二(state)
	state = pinLeavesDimensions0零一(state)
	workersMaximum: int = defineProcessorLimit(CPUlimit)
	print(doTheNeedful(state, workersMaximum).foldsTotal)

"""
