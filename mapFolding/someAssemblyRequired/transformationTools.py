"""
Tools for transforming Python code through abstract syntax tree (AST) manipulation.

This module provides a comprehensive set of utilities for programmatically analyzing,
transforming, and generating Python code through AST manipulation. It implements
a highly flexible framework that enables:

1. Precise identification of code patterns through composable predicates
2. Targeted modification of code structures while preserving semantics
3. Code generation with proper syntax and import management
4. Analysis of code dependencies and relationships
5. Clean transformation of one algorithmic implementation to another

The utilities are organized into several key components:
- Predicate factories (ifThis): Create composable functions for matching AST patterns
- Node transformers: Modify AST structures in targeted ways
- Code generation helpers (Make): Create well-formed AST nodes programmatically
- Import tracking: Maintain proper imports during code transformation
- Analysis tools: Extract and organize code information

While these tools were developed to transform the baseline algorithm into optimized formats,
they are designed as general-purpose utilities applicable to a wide range of code
transformation scenarios beyond the scope of this package.
"""
from collections.abc import Callable, Sequence
from copy import deepcopy
from mapFolding.someAssemblyRequired import ast_Identifier, ifThis, nodeType, Then
from typing import Any, cast, Generic, TypeGuard
import ast

"""
Semiotic notes:
In the `ast` package, some things that look and feel like a "name" are not `ast.Name` type. The following semiotics are a balance between technical precision and practical usage.

astName: always means `ast.Name`.
Name: uppercase, _should_ be interchangeable with astName, even in camelCase.
Hunter: ^^ did you do that ^^ ? Are you sure? You just fixed some that should have been "_name" because it confused you.
name: lowercase, never means `ast.Name`. In camelCase, I _should_ avoid using it in such a way that it could be confused with "Name", uppercase.
_Identifier: very strongly correlates with the private `ast._Identifier`, which is a TypeAlias for `str`.
identifier: lowercase, a general term that includes the above and other Python identifiers.
Identifier: uppercase, without the leading underscore should only appear in camelCase and means "identifier", lowercase.
namespace: lowercase, in dotted-names, such as `pathlib.Path` or `collections.abc`, "namespace" is the part before the dot.
Namespace: uppercase, should only appear in camelCase and means "namespace", lowercase.
"""

# Would `LibCST` be better than `ast` in some cases? https://github.com/hunterhogan/mapFolding/issues/7

class NodeCollector(Generic[nodeType], ast.NodeVisitor):
	"""A node visitor that collects data via one or more actions when a predicate is met."""
	def __init__(self, findThis: Callable[[ast.AST], TypeGuard[nodeType] | bool], doThat: list[Callable[[nodeType], Any]]) -> None:
		self.findThis = findThis
		self.doThat = doThat

	def visit(self, node: ast.AST) -> None:
		if self.findThis(node):
			for action in self.doThat:
				action(cast(nodeType, node))
		self.generic_visit(node)

class NodeReplacer(Generic[nodeType], ast.NodeTransformer):
	"""A node transformer that replaces or removes AST nodes based on a condition."""
	def __init__(self, findThis: Callable[[ast.AST], TypeGuard[nodeType] | bool], doThat: Callable[[nodeType], ast.AST | Sequence[ast.AST] | None]) -> None:
		self.findThis = findThis
		self.doThat = doThat

	def visit(self, node: ast.AST) -> ast.AST | Sequence[ast.AST] | None:
		if self.findThis(node):
			return self.doThat(cast(nodeType, node))
		return super().visit(node)

def extractClassDef(module: ast.Module, identifier: ast_Identifier) -> ast.ClassDef | None:
	sherpa: list[ast.ClassDef] = []
	extractor = NodeCollector(ifThis.isClassDef_Identifier(identifier), [Then.appendTo(sherpa)])
	extractor.visit(module)
	astClassDef = sherpa[0] if sherpa else None
	return astClassDef

def extractFunctionDef(module: ast.Module, identifier: ast_Identifier) -> ast.FunctionDef | None:
	sherpa: list[ast.FunctionDef] = []
	extractor = NodeCollector(ifThis.isFunctionDef_Identifier(identifier), [Then.appendTo(sherpa)])
	extractor.visit(module)
	astClassDef = sherpa[0] if sherpa else None
	return astClassDef

def makeDictionaryFunctionDef(module: ast.Module) -> dict[ast_Identifier, ast.FunctionDef]:
	dictionaryFunctionDef: dict[ast_Identifier, ast.FunctionDef] = {}
	NodeCollector(ifThis.isFunctionDef, [Then.updateThis(dictionaryFunctionDef)]).visit(module)
	return dictionaryFunctionDef

def makeDictionaryReplacementStatements(module: ast.Module) -> dict[ast_Identifier, ast.stmt | list[ast.stmt]]:
	"""Return a dictionary of function names and their replacement statements."""
	dictionaryFunctionDef: dict[ast_Identifier, ast.FunctionDef] = makeDictionaryFunctionDef(module)
	dictionaryReplacementStatements: dict[ast_Identifier, ast.stmt | list[ast.stmt]] = {}
	for name, astFunctionDef in dictionaryFunctionDef.items():
		if ifThis.onlyReturnAnyCompare(astFunctionDef):
			dictionaryReplacementStatements[name] = astFunctionDef.body[0].value # type: ignore
		elif ifThis.onlyReturnUnaryOp(astFunctionDef):
			dictionaryReplacementStatements[name] = astFunctionDef.body[0].value # type: ignore
		else:
			dictionaryReplacementStatements[name] = astFunctionDef.body[0:-1]
	return dictionaryReplacementStatements

def Z0Z_descendantContainsMatchingNode(node: ast.AST, predicateFunction: Callable[[ast.AST], bool]) -> bool:
	"""Return True if any descendant of the node (or the node itself) matches the predicateFunction."""
	matchFound = False

	class DescendantFinder(ast.NodeVisitor):
		def generic_visit(self, node: ast.AST) -> None:
			nonlocal matchFound
			if predicateFunction(node):
				matchFound = True
			else:
				super().generic_visit(node)

	DescendantFinder().visit(node)
	return matchFound

def Z0Z_executeActionUnlessDescendantMatches(exclusionPredicate: Callable[[ast.AST], bool], actionFunction: Callable[[ast.AST], None]) -> Callable[[ast.AST], None]:
	"""Return a new action that will execute actionFunction only if no descendant (or the node itself) matches exclusionPredicate."""
	def wrappedAction(node: ast.AST) -> None:
		if not Z0Z_descendantContainsMatchingNode(node, exclusionPredicate):
			actionFunction(node)
	return wrappedAction

def inlineThisFunctionWithTheseValues(astFunctionDef: ast.FunctionDef, dictionaryReplacementStatements: dict[str, ast.stmt | list[ast.stmt]]) -> ast.FunctionDef:
	class FunctionInliner(ast.NodeTransformer):
		def __init__(self, dictionaryReplacementStatements: dict[str, ast.stmt | list[ast.stmt]]) -> None:
			self.dictionaryReplacementStatements = dictionaryReplacementStatements

		def generic_visit(self, node: ast.AST) -> ast.AST:
			"""Visit all nodes and replace them if necessary."""
			return super().generic_visit(node)

		def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.stmt]:
			if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node.value):
				return self.dictionaryReplacementStatements[node.value.func.id] # type: ignore[attr-defined]
			return node

		def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
			if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node.value):
				return self.dictionaryReplacementStatements[node.value.func.id] # type: ignore[attr-defined]
			return node

		def visit_Call(self, node: ast.Call) -> ast.AST | list[ast.stmt]:
			if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node):
				replacement = self.dictionaryReplacementStatements[node.func.id] # type: ignore[attr-defined]
				if not isinstance(replacement, list):
					return replacement
			return node

	keepGoing = True
	ImaInlineFunction = deepcopy(astFunctionDef)
	while keepGoing:
		ImaInlineFunction = deepcopy(astFunctionDef)
		FunctionInliner(deepcopy(dictionaryReplacementStatements)).visit(ImaInlineFunction)
		if ast.unparse(ImaInlineFunction) == ast.unparse(astFunctionDef):
			keepGoing = False
		else:
			astFunctionDef = deepcopy(ImaInlineFunction)
			ast.fix_missing_locations(astFunctionDef)
	return ImaInlineFunction

def Z0Z_replaceMatchingASTnodes(astTree: ast.AST, mappingFindReplaceNodes: dict[ast.AST, ast.AST]) -> ast.AST:
	class TargetedNodeReplacer(ast.NodeTransformer):
		def __init__(self, mappingFindReplaceNodes: dict[ast.AST, ast.AST]) -> None:
			self.mappingFindReplaceNodes = mappingFindReplaceNodes

		def visit(self, node: ast.AST) -> ast.AST:
			for nodeFind, nodeReplace in self.mappingFindReplaceNodes.items():
				if self.nodesMatchStructurally(node, nodeFind):
					return nodeReplace
			return self.generic_visit(node)

		def nodesMatchStructurally(self, nodeSubject: ast.AST | list[Any] | Any, nodePattern: ast.AST | list[Any] | Any) -> bool:
			if nodeSubject is None or nodePattern is None:
				return nodeSubject is None and nodePattern is None

			if type(nodeSubject) is not type(nodePattern):
				return False

			if isinstance(nodeSubject, ast.AST):
				for field, fieldValueSubject in ast.iter_fields(nodeSubject):
					if field in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset', 'ctx'):
						continue
					attrPattern = getattr(nodePattern, field, None)
					if not self.nodesMatchStructurally(fieldValueSubject, attrPattern):
						return False
				return True

			if isinstance(nodeSubject, list) and isinstance(nodePattern, list):
				nodeSubjectList: list[Any] = nodeSubject
				nodePatternList: list[Any] = nodePattern
				return len(nodeSubjectList) == len(nodePatternList) and all(
					self.nodesMatchStructurally(elementSubject, elementPattern)
					for elementSubject, elementPattern in zip(nodeSubjectList, nodePatternList)
				)

			return nodeSubject == nodePattern

	astTreeCurrent, astTreePrevious = None, astTree
	while astTreeCurrent is None or ast.unparse(astTreeCurrent) != ast.unparse(astTreePrevious):
		astTreePrevious = astTreeCurrent if astTreeCurrent else astTree
		astTreeCurrent = TargetedNodeReplacer(mappingFindReplaceNodes).visit(astTreePrevious)

	return astTreeCurrent
