"""The containers, transformations, predicates, and paths that the module generators in `mapFolding.kitAST` compose.

A defect in one of these reaches every synthesized module, so each target is tested here, before it is nine steps removed from a
malformed module. Expected values come from the targets' docstrings, hand-derived, unless a test's docstring names another
basis. `ast.unparse` formats every expected source.
"""
from __future__ import annotations

from astToolkit import Make
from collections import Counter
from mapFolding.kitAST import (
	astModuleToIngredientsFunction, IfThis, IngredientsFunction, IngredientsModule, LedgerOfImports, removeFunctionDef, renameFunctionDef,
	renameName)
from mapFolding.kitAST.paths import getLogicalPath, makePathFilenameModule
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.sourcesPython import sourceFunctionsChicken, sourceImportsTopLevel
from pathlib import PurePath
from typing import TYPE_CHECKING
import ast
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterMakesPy import CallableFunction
	from typing import TypeIs

@pytest.mark.parametrize('source, expected', (
	pytest.param(sourceImportsTopLevel, Counter(('from __future__ import annotations', 'from collections import deque'
		, 'from typing import TYPE_CHECKING', 'import numpy', 'import tomllib', 'import math', 'if TYPE_CHECKING:\n    from pathlib import Path'))
		, id='sourceImportsTopLevel'),
	pytest.param('from collections import deque\nfrom collections import Counter\nfrom collections import deque\nimport numpy\nimport numpy\n'
		'if TYPE_CHECKING:\n    import numpy\n    from pathlib import Path', Counter(('from collections import Counter, deque', 'import numpy'
		, 'from typing import TYPE_CHECKING', 'if TYPE_CHECKING:\n    from pathlib import Path')), id='importsRepeated'),
))
def test_LedgerOfImportsMakeList_ast(source: str, expected: Counter[str]) -> None:
	"""The ledger records every import in `source`, at any depth; each recorded import appears in exactly one statement, a type-only
	import that is also a runtime import appears only at runtime, and the `TYPE_CHECKING` block arrives with its own import. The
	docstring states no order between statements, so the statements are compared as a collection.
	"""
	assertEqualTo(Counter(map(ast.unparse, LedgerOfImports(ast.parse(source)).makeList_ast())), expected, LedgerOfImports.makeList_ast.__name__, source)

@pytest.mark.parametrize('source, expectedBody, expectedImports', (
	pytest.param(sourceImportsTopLevel, 'if sys.version_info >= (3, 14):\n    import tomllib\n\ndef chicken(egg: Path):\n    import math\n'
		'    return deque((math.pi, egg))', Counter(('from __future__ import annotations', 'from collections import deque'
		, 'from typing import TYPE_CHECKING', 'import numpy', 'if TYPE_CHECKING:\n    from pathlib import Path')), id='sourceImportsTopLevel'),
	pytest.param('import numpy\nif TYPE_CHECKING:\n    from pathlib import Path\n    Chicken = Path', 'if TYPE_CHECKING:\n'
		'    from pathlib import Path\n    Chicken = Path', Counter(('import numpy',)), id='typeCheckingWithAssignment'),
))
def test_LedgerOfImportsExtractImports(source: str, expectedBody: str, expectedImports: Counter[str]) -> None:
	"""The module keeps every statement except the top-level imports and import-only `TYPE_CHECKING` blocks, which the ledger records."""
	astModule: ast.Module = ast.parse(source)
	ledger = LedgerOfImports()
	ledger.extractImports(astModule)
	assertEqualTo(ast.unparse(astModule), expectedBody, LedgerOfImports.extractImports.__name__, source)
	assertEqualTo(Counter(map(ast.unparse, ledger.makeList_ast())), expectedImports, LedgerOfImports.extractImports.__name__, source)

@pytest.mark.parametrize('identifier, expectedFunctionDef, expectedImports', (
	pytest.param('chicken', 'def chicken(egg: Path):\n    import math\n    return deque((math.pi, egg))', Counter((
		'from __future__ import annotations', 'from collections import deque', 'from typing import TYPE_CHECKING', 'import numpy'
		, 'import tomllib', 'import math', 'if TYPE_CHECKING:\n    from pathlib import Path')), id='chicken'),
))
def test_astModuleToIngredientsFunction(identifier: str, expectedFunctionDef: str, expectedImports: Counter[str]) -> None:
	"""The function arrives with every import of `sourceImportsTopLevel`, wherever the module declares it."""
	ingredientsFunction: IngredientsFunction = astModuleToIngredientsFunction(ast.parse(sourceImportsTopLevel), identifier)
	assertEqualTo(ast.unparse(ingredientsFunction.astFunctionDef), expectedFunctionDef, astModuleToIngredientsFunction.__name__, identifier)
	assertEqualTo(Counter(map(ast.unparse, ingredientsFunction.imports.makeList_ast())), expectedImports
		, astModuleToIngredientsFunction.__name__, identifier)

@pytest.mark.parametrize('launcher, expected', (
	pytest.param('print(13)', "if __name__ == '__main__':\n    print(13)", id='launcherBare'),
	pytest.param("if __name__ == '__main__':\n    print(13)", "if __name__ == '__main__':\n    print(13)", id='launcherGuarded'),
	pytest.param("if __name__ == '__main__':\n    if egg:\n        print(13)", "if __name__ == '__main__':\n    if egg:\n        print(13)"
		, id='launcherGuardedConditional'),
))
def test_IngredientsModuleBody(launcher: str, expected: str) -> None:
	"""The body holds the imports, the prologue, the functions, the epilogue, and the launcher inside one `__main__` guard, in order."""
	ingredientsFunction = IngredientsFunction(Make.FunctionDef('hen', body=[Make.Return(Make.Name('chicken'))]))
	ingredientsFunction.imports.addImport_asStr('numpy')
	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.appendPrologue(ast.parse('chicken = 13'))
	ingredientsModule.appendEpilogue(ast.parse('rooster = hen()'))
	ingredientsModule.appendLauncher(ast.parse(launcher))
	assertEqualTo(ast.unparse(ast.fix_missing_locations(Make.Module(ingredientsModule.body))), f'import numpy\nchicken = 13\n\ndef hen():\n    return chicken\nrooster = hen()\n{expected}'
		, 'IngredientsModule.body', launcher)

@pytest.mark.parametrize('source', (
	pytest.param('chicken = 13  # type: ignore[assignment]', marks=pytest.mark.xfail(strict=True
		, reason='`IngredientsModule.type_ignores` extends `_supplemental_type_ignores` in place, so each read adds the directives again.')
		, id='typeIgnoreAssignment'),
))
def test_IngredientsModuleTypeIgnores(source: str) -> None:
	"""Relation: reading `type_ignores` again returns the same directives."""
	ingredientsModule = IngredientsModule()
	ingredientsModule.appendPrologue(ast.parse(source, type_comments=True))
	assertEqualTo(len(ingredientsModule.type_ignores), len(ingredientsModule.type_ignores), 'IngredientsModule.type_ignores', source)

@pytest.mark.parametrize('identifier, expected', (
	pytest.param('chicken', 'def coop():\n    return chicken', id='chicken'),
	pytest.param('rooster', sourceFunctionsChicken, id='rooster'),
))
def test_removeFunctionDef(identifier: str, expected: str) -> None:
	astModule: ast.Module = ast.parse(sourceFunctionsChicken)
	removeFunctionDef(identifier, astModule)
	assertEqualTo(ast.unparse(astModule), expected, removeFunctionDef.__name__, identifier)

@pytest.mark.parametrize('identifier, identifierNew, expected', (
	pytest.param('chicken', 'hen', 'def hen(egg):\n    return chicken.feathers + rooster.chicken + egg\n\ndef coop():\n\n    def hen():\n'
		'        return 13\n    return chicken', id='chicken-hen'),
))
def test_renameFunctionDef(identifier: str, identifierNew: str, expected: str) -> None:
	astModule: ast.Module = ast.parse(sourceFunctionsChicken)
	renameFunctionDef(identifier, identifierNew, astModule)
	assertEqualTo(ast.unparse(astModule), expected, renameFunctionDef.__name__, identifier, identifierNew)

@pytest.mark.parametrize('identifier, identifierNew, expected', (
	pytest.param('chicken', 'hen', 'def chicken(egg):\n    return hen.feathers + rooster.chicken + egg\n\ndef coop():\n\n    def chicken():\n'
		'        return 13\n    return hen', id='chicken-hen'),
))
def test_renameName(identifier: str, identifierNew: str, expected: str) -> None:
	astModule: ast.Module = ast.parse(sourceFunctionsChicken)
	renameName(identifier, identifierNew, astModule)
	assertEqualTo(ast.unparse(astModule), expected, renameName.__name__, identifier, identifierNew)

@pytest.mark.parametrize('identifierPackage, logicalPathInfix, identifierModule, expected', (
	pytest.param('mapFolding', 'kitAST.numba', ('makeJob',), 'mapFolding.kitAST.numba.makeJob', id='packageInfixModule'),
	pytest.param(None, None, ('chicken',), 'chicken', id='moduleOnly'),
	pytest.param('mapFolding', '', ('synthesized', 'theorem2'), 'mapFolding.synthesized.theorem2', id='infixEmpty'),
))
def test_getLogicalPath(identifierPackage: str | None, logicalPathInfix: str | None, identifierModule: tuple[str, ...], expected: str) -> None:
	assertEqualTo(getLogicalPath(identifierPackage, logicalPathInfix, *identifierModule), expected, getLogicalPath.__name__
		, identifierPackage, logicalPathInfix, *identifierModule)

@pytest.mark.parametrize('pathRoot, logicalPathInfix, identifierModule, fileExtension, expected', (
	pytest.param(None, None, 'chicken', '.py', PurePath('chicken.py'), id='moduleOnly'),
	pytest.param(PurePath('coop'), 'synthesized.matrixMeanders', 'prune', '.py', PurePath('coop/synthesized/matrixMeanders/prune.py'), id='rootInfix'),
	pytest.param(PurePath('coop'), None, 'jobNumba', '.pyi', PurePath('coop/jobNumba.pyi'), id='rootStub'),
))
def test_getPathFilename(pathRoot: PurePath | None, logicalPathInfix: str | None, identifierModule: str, fileExtension: str, expected: PurePath) -> None:
	assertEqualTo(makePathFilenameModule(pathRoot, logicalPathInfix, identifierModule, fileExtension), expected, makePathFilenameModule.__name__
		, pathRoot, logicalPathInfix, identifierModule, fileExtension)

@pytest.mark.parametrize('predicate, source, expected', (
	pytest.param(IfThis.is0LessThanAttributeNamespaceIdentifier, '0 < state.leaf', True, id='is0LessThan-zeroLessThanLeaf'),
	pytest.param(IfThis.is0LessThanAttributeNamespaceIdentifier, 'state.leaf > 0', False, id='is0LessThan-leafGreaterThanZero'),
	pytest.param(IfThis.is0LessThanAttributeNamespaceIdentifier, '0 < state.pile', False, id='is0LessThan-zeroLessThanPile'),
	pytest.param(IfThis.is0LessThanAttributeNamespaceIdentifier, '0 <= state.leaf', False, id='is0LessThan-zeroLessThanOrEqualLeaf'),
	pytest.param(IfThis.isAttributeNamespaceIdentifierGreaterThan0, 'state.leaf > 0', True, id='isGreaterThan0-leafGreaterThanZero'),
	pytest.param(IfThis.isAttributeNamespaceIdentifierGreaterThan0, '0 < state.leaf', False, id='isGreaterThan0-zeroLessThanLeaf'),
	pytest.param(IfThis.isAttributeNamespaceIdentifierGreaterThan0, 'state.leaf > 1', False, id='isGreaterThan0-leafGreaterThanOne'),
	pytest.param(IfThis.isAttributeNamespaceIdentifierGreaterThan0, 'leaf.state > 0', False, id='isGreaterThan0-namespaceSwapped'),
	pytest.param(IfThis.isAttributeNamespaceIdentifierLessThanOrEqual0, 'state.leaf <= 0', True, id='isLessThanOrEqual0-leafLessThanOrEqualZero'),
	pytest.param(IfThis.isAttributeNamespaceIdentifierLessThanOrEqual0, 'state.leaf < 0', False, id='isLessThanOrEqual0-leafLessThanZero'),
	pytest.param(IfThis.isAttributeNamespaceIdentifierLessThanOrEqual0, '0 >= state.leaf', False, id='isLessThanOrEqual0-zeroGreaterThanOrEqualLeaf'),
))
def test_IfThisCompare(predicate: CallableFunction[[str, str], Callable[[ast.AST], TypeIs[ast.Compare]]], source: str, expected: bool) -> None:
	"""`isAttributeNamespaceIdentifierLessThanOrEqual0` documents its left operand and first operator, and no comparator."""
	assert predicate('state', 'leaf')(ast.parse(source, mode='eval').body) is expected, f'{predicate.__name__}(state, leaf)({source!r}) is not {expected}.'

@pytest.mark.parametrize('source', (
	pytest.param('if 0 < state.leaf:\n    pass', id='ifZeroLessThanLeaf'),
	pytest.param('if state.leaf > 0:\n    pass', id='ifLeafGreaterThanZero'),
	pytest.param('while 0 < state.leaf:\n    pass', id='whileZeroLessThanLeaf'),
	pytest.param('while state.leaf > 0:\n    pass', id='whileLeafGreaterThanZero'),
))
@pytest.mark.parametrize('predicate, sourceMatching', (
	pytest.param(IfThis.isIf0LessThanAttributeNamespaceIdentifier, 'if 0 < state.leaf:\n    pass', id='isIf0LessThan'),
	pytest.param(IfThis.isIfAttributeNamespaceIdentifierGreaterThan0, 'if state.leaf > 0:\n    pass', id='isIfGreaterThan0'),
	pytest.param(IfThis.isWhile0LessThanAttributeNamespaceIdentifier, 'while 0 < state.leaf:\n    pass', id='isWhile0LessThan'),
	pytest.param(IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0, 'while state.leaf > 0:\n    pass', id='isWhileGreaterThan0'),
))
def test_IfThisStatement(predicate: CallableFunction[[str, str], Callable[[ast.AST], bool]], sourceMatching: str, source: str) -> None:
	"""Each predicate matches the statement of its own kind and comparison, and none of the others."""
	assert predicate('state', 'leaf')(ast.parse(source).body[0]) is (source == sourceMatching), f'{predicate.__name__}(state, leaf)({source!r})'
