from __future__ import annotations

from anyascii import anyascii
from ast import parse as ast_parse
from astToolkit import Be, IfThis, Make, NodeChanger, NodeTourist, parseLogicalPath2astModule, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsModule, LedgerOfImports
from humpy_cytoolz import juxt
from itertools import repeat, starmap
from mapFolding._e.kitAST.theSSOT import default
from mapFolding.kitAST.linux import toCodon
from mapFolding.theSSOT import settingsPackage
from pathlib import Path
from typing import TYPE_CHECKING
import autoflake  # pyright: ignore[reportMissingTypeStubs] # TODO waiting for new version.
import python_minifier

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterMakesPy import identifierDotAttribute
	from typing import TypeIs
	import ast

launcher = """
# ruff: file-ignore[commented-out-code, print]
if __name__ == "__main__":
	CPUlimit: int | float | None = None
	state: StateElimination = StateElimination((2,) * 5)
	# state = pinPile零Ante首零(state)
	state = pinPilesAtEnds(state, 3)
	state = pinLeavesDimension首二(state)
	# state = pin3beans2(state)
	# state = pin首beans(state)
	# state = pinLeavesDimension一(state)
	# state = pinLeavesDimension二(state)
	state = pinLeavesDimensions0零一(state)
	workersMaximum: int = defineProcessorLimit(CPUlimit)
	print(doTheNeedful(state, workersMaximum).totalFolds)

"""

def assimilateEliminationCrease(identifierModuleBorg: str) -> Path:
	def assimilateFunction(logicalPathAssimilee: identifierDotAttribute, identifierFunction: str) -> None:
		moduleBorg.appendIngredientsFunction(astModuleToIngredientsFunction(dissectModule(logicalPathAssimilee), identifierFunction))

	def assimilateModule(logicalPathAssimilee: identifierDotAttribute) -> None:
		moduleAssimilee: ast.Module = dissectModule(logicalPathAssimilee)

		moduleBorg.imports.walkThis(moduleAssimilee)
		NodeChanger(IfThis.isAnyOf(Be.Import, Be.ImportFrom), Then.removeIt).visit(moduleAssimilee)

		moduleBorg.appendEpilogue(moduleAssimilee)

	def dissectModule(logicalPathAssimilee: identifierDotAttribute) -> ast.Module:
		moduleDissect: ast.Module = parseLogicalPath2astModule(logicalPathAssimilee, optimize=2)

		# Remove docstrings that ast.parse didn't think were docstrings.
		NodeChanger(Be.Expr.valueIs(Be.Constant.valueIs(lambda node: isinstance(node, str))), Then.removeIt).visit(moduleDissect)

		findThis: Callable[[ast.AST], TypeIs[ast.If]] = Be.If.testIs(Be.Name.idIs('TYPE_CHECKING'.__eq__))
		NodeTourist(findThis, ledgerTYPE_CHECKING.walkThis).visit(moduleDissect)
		NodeChanger(findThis, Then.removeIt).visit(moduleDissect)

		return moduleDissect

	boxOfFunctionsHARDCODED: list[tuple[identifierDotAttribute, str]] = [('mapFolding.beDRY', 'getTotalLeaves'), ('mapFolding.beDRY', 'defineProcessorLimit')]
	boxOfModulesHARDCODED: list[identifierDotAttribute] = [
		*tuple(map("{0}._e.{1}".format, repeat(settingsPackage.identifierPackage), (
			'theTypes', 'semiotics', 'leafDomains', 'pileOptions', '_disaggregation', '_beDRY', 'dataBaskets', 'filters', 'pinIt'
		)))
		, *tuple(map("{0}._e._2上nDimensional.{1}".format, repeat(settingsPackage.identifierPackage), (
			'pinIt', 'pinByCrease', 'pinByDomain', 'pinItAnnex', 'semiotics', 'beDRY', 'measure', 'creases'
			, 'leafDomains', 'conditionalOrdering', 'pileOptions', 'filters'
		)))
		, f"{default['logicalPath']['algorithm']}.iff"
		, f"{default['logicalPath']['algorithm']}.{default['module']['algorithm']}"
	]
	boxOfPackagesHARDCODED: list[identifierDotAttribute] = [*tuple(map("{0}.{1}".format, repeat(settingsPackage.identifierPackage), ('beDRY', '_e', '_e._2上nDimensional')))]

	boxOfFunctions: list[tuple[identifierDotAttribute, str]] = boxOfFunctionsHARDCODED
	boxOfModules: list[identifierDotAttribute] = boxOfModulesHARDCODED
	boxOfPackages: list[identifierDotAttribute] = [*boxOfPackagesHARDCODED, *boxOfModules]

	ledgerTYPE_CHECKING = LedgerOfImports()
	moduleBorg = IngredientsModule()

	tuple(starmap(assimilateFunction, boxOfFunctions))
	tuple(map(assimilateModule, boxOfModules))

	tuple(map(juxt(ledgerTYPE_CHECKING.removeImportFrom, moduleBorg.removeImportFrom), boxOfPackages, repeat(None)))

	moduleBorg.appendPrologue(statement=Make.If(Make.Name('TYPE_CHECKING'), ledgerTYPE_CHECKING.makeList_ast()))
	moduleBorg.appendLauncher(ast_parse(launcher))

	pathFilename = Path(*default['logicalPath']['synthetic'].split('.'), identifierModuleBorg + settingsPackage.fileExtension)
	return moduleBorg.write_astModule(pathFilename, settingsPackage.identifierPackage)

# TODO If I keep this functionality, do the disk i/o with an appropriate function.
def minify(pathFilename: Path) -> Path:
	pathFilename.with_stem('min').write_text(python_minifier.minify(autoflake.fix_code(pathFilename.read_text(encoding='utf-8'), remove_unused_variables=True)
		, remove_literal_statements=True, rename_globals=True, prefer_single_line=False), encoding='utf-8')
	return pathFilename.with_stem('min')

# TODO If I keep this functionality, do the disk i/o with an appropriate function.
def toASCII(pathFilename: Path) -> Path:
	pathFilename.with_stem('ascii').write_text(anyascii(pathFilename.read_text(encoding='utf-8')).replace(', /', '').replace(', *,', ','), encoding='ascii')
	return pathFilename.with_stem('ascii')

if __name__ == '__main__':
	toCodon(minify(assimilateEliminationCrease('module1')))
	toCodon(toASCII(assimilateEliminationCrease('module1')))
	pathFilename = settingsPackage.pathPackage / '_e' / 'kitAST' / 'aa.py'
	toCodon(pathFilename)
