"""
AST Node Predicate and Access Utilities for Pattern Matching and Traversal

This module provides utilities for accessing and matching AST nodes in a consistent way. It contains three primary
classes:

1. DOT: Provides consistent accessor methods for AST node attributes across different node types, simplifying the access
	to node properties.

2. be: Offers type-guard functions that verify AST node types, enabling safe type narrowing for static type checking and
	improving code safety.

3. ifThis: Contains predicate functions for matching AST nodes based on various criteria, enabling precise targeting of
	nodes for analysis or transformation.

These utilities form the foundation of the pattern-matching component in the AST manipulation framework, working in
conjunction with the NodeChanger and NodeTourist classes to enable precise and targeted code transformations. Together,
they implement a declarative approach to AST manipulation that separates node identification (ifThis), type verification
(be), and data access (DOT).
"""

from collections.abc import Callable
from mapFolding.someAssemblyRequired import (
	ast_Identifier,
	be,
	DOT,
	ImaCallToName,
)
from typing import Any, TypeGuard
import ast

class ifThis:
	"""
	Provide predicate functions for matching and filtering AST nodes based on various criteria.

	The ifThis class contains static methods that generate predicate functions used to test whether AST nodes match
	specific criteria. These predicates can be used with NodeChanger and NodeTourist to identify and process specific
	patterns in the AST.

	The class provides predicates for matching various node types, attributes, identifiers, and structural patterns,
	enabling precise targeting of AST elements for analysis or transformation.
	"""
	@staticmethod
	def _Identifier(identifier: ast_Identifier) -> Callable[[ast_Identifier | None], TypeGuard[ast_Identifier] | bool]:
		return lambda node: node == identifier
	@staticmethod
	def _nested_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute | ast.Starred | ast.Subscript] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.Attribute | ast.Starred | ast.Subscript] | bool:
			return ifThis.isName_Identifier(identifier)(node) or ifThis.isAttribute_Identifier(identifier)(node) or ifThis.isSubscript_Identifier(identifier)(node) or ifThis.isStarred_Identifier(identifier)(node)
		return workhorse

	@staticmethod
	def is_arg_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.arg] | bool]:
		"""see also `isArgument_Identifier`"""
		return lambda node: be.arg(node) and ifThis._Identifier(identifier)(DOT.arg(node))
	@staticmethod
	def is_keyword_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.keyword] | bool]:
		"""see also `isArgument_Identifier`"""
		return lambda node: be.keyword(node) and ifThis._Identifier(identifier)(DOT.arg(node))

	@staticmethod
	def isAnnAssign_targetIs(targetPredicate: Callable[[ast.expr], TypeGuard[ast.expr] | bool]) -> Callable[[ast.AST], TypeGuard[ast.AnnAssign] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.AnnAssign] | bool:
			return be.AnnAssign(node) and targetPredicate(DOT.target(node))
		return workhorse

	@staticmethod
	def isArgument_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.arg | ast.keyword] | bool]:
		return lambda node: (be.arg(node) or be.keyword(node)) and ifThis._Identifier(identifier)(DOT.arg(node))

	@staticmethod
	def isAssignAndTargets0Is(targets0Predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], TypeGuard[ast.AnnAssign] | bool]:
		"""node is Assign and node.targets[0] matches `targets0Predicate`."""
		return lambda node: be.Assign(node) and targets0Predicate(node.targets[0])
	@staticmethod
	def isAssignAndValueIs(valuePredicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], TypeGuard[ast.Assign] | bool]:
		"""node is ast.Assign and node.value matches `valuePredicate`. """
		return lambda node: be.Assign(node) and valuePredicate(DOT.value(node))

	@staticmethod
	def isAttribute_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute] | bool]:
		"""node is `ast.Attribute` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Attribute]:
			return be.Attribute(node) and ifThis._nested_Identifier(identifier)(DOT.value(node))
		return workhorse
	@staticmethod
	def isAttributeName(node: ast.AST) -> TypeGuard[ast.Attribute]:
		""" Displayed as Name.attribute."""
		return be.Attribute(node) and be.Name(DOT.value(node))
	@staticmethod
	def isAttributeNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Attribute] | bool]:
		return lambda node: ifThis.isAttributeName(node) and ifThis.isName_Identifier(namespace)(DOT.value(node)) and ifThis._Identifier(identifier)(DOT.attr(node))

	@staticmethod
	def isAttributeNamespace_IdentifierGreaterThan0(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Compare] | bool]:
		return lambda node: (be.Compare(node)
					and ifThis.isAttributeNamespace_Identifier(namespace, identifier)(node.left)
					and be.Gt(node.ops[0])
					and ifThis.isConstant_value(0)(node.comparators[0]))

	@staticmethod
	def isUnaryNotAttributeNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.UnaryOp] | bool]:
		return lambda node: (be.UnaryOp(node)
					and be.Not(node.op)
					and ifThis.isAttributeNamespace_Identifier(namespace, identifier)(node.operand))

	@staticmethod
	def isIfUnaryNotAttributeNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.If] | bool]:
		return lambda node: (be.If(node)
					and ifThis.isUnaryNotAttributeNamespace_Identifier(namespace, identifier)(node.test))

	@staticmethod
	def isIfAttributeNamespace_IdentifierGreaterThan0(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.If] | bool]:
		return lambda node: (be.If(node)
					and ifThis.isAttributeNamespace_IdentifierGreaterThan0(namespace, identifier)(node.test))

	@staticmethod
	def isWhileAttributeNamespace_IdentifierGreaterThan0(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.While] | bool]:
		return lambda node: (be.While(node)
					and ifThis.isAttributeNamespace_IdentifierGreaterThan0(namespace, identifier)(node.test))

	@staticmethod
	def isAttributeNamespace_IdentifierLessThanOrEqual(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Compare] | bool]:
		return lambda node: (be.Compare(node)
					and ifThis.isAttributeNamespace_Identifier(namespace, identifier)(node.left)
					and be.LtE(node.ops[0]))

	@staticmethod
	def isAugAssign_targetIs(targetPredicate: Callable[[ast.expr], TypeGuard[ast.expr] | bool]) -> Callable[[ast.AST], TypeGuard[ast.AugAssign] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.AugAssign] | bool:
			return be.AugAssign(node) and targetPredicate(DOT.target(node))
		return workhorse

	@staticmethod
	def isCall_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ImaCallToName] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ImaCallToName] | bool:
			return ifThis.isCallToName(node) and ifThis._Identifier(identifier)(DOT.id(DOT.func(node)))
		return workhorse

	@staticmethod
	def isCallAttributeNamespace_Identifier(namespace: ast_Identifier, identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Call] | bool]:
		def workhorse(node: ast.AST) -> TypeGuard[ast.Call] | bool:
			return be.Call(node) and ifThis.isAttributeNamespace_Identifier(namespace, identifier)(DOT.func(node))
		return workhorse
	@staticmethod
	def isCallToName(node: ast.AST) -> TypeGuard[ImaCallToName]:
		return be.Call(node) and be.Name(DOT.func(node))

	@staticmethod
	def isClassDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.ClassDef] | bool]:
		return lambda node: be.ClassDef(node) and ifThis._Identifier(identifier)(DOT.name(node))

	@staticmethod
	def isConstant_value(value: Any) -> Callable[[ast.AST], TypeGuard[ast.Constant] | bool]:
		return lambda node: be.Constant(node) and DOT.value(node) == value

	@staticmethod
	def isFunctionDef_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.FunctionDef] | bool]:
		return lambda node: be.FunctionDef(node) and ifThis._Identifier(identifier)(DOT.name(node))

	@staticmethod
	def isName_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Name] | bool]:
		return lambda node: be.Name(node) and ifThis._Identifier(identifier)(DOT.id(node))

	@staticmethod
	def isStarred_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Starred] | bool]:
		"""node is `ast.Starred` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Starred]:
			return be.Starred(node) and ifThis._nested_Identifier(identifier)(DOT.value(node))
		return workhorse
	@staticmethod
	def isSubscript_Identifier(identifier: ast_Identifier) -> Callable[[ast.AST], TypeGuard[ast.Subscript] | bool]:
		"""node is `ast.Subscript` and the top-level `ast.Name` is `identifier`"""
		def workhorse(node: ast.AST) -> TypeGuard[ast.Subscript]:
			return be.Subscript(node) and ifThis._nested_Identifier(identifier)(DOT.value(node))
		return workhorse

	@staticmethod
	def matchesMeButNotAnyDescendant(predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		return lambda node: predicate(node) and ifThis.matchesNoDescendant(predicate)(node)
	@staticmethod
	def matchesNoDescendant(predicate: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		def workhorse(node: ast.AST) -> bool:
			for descendant in ast.walk(node):
				if descendant is not node and predicate(descendant):
					return False
			return True
		return workhorse

	@staticmethod
	def Z0Z_unparseIs(astAST: ast.AST) -> Callable[[ast.AST], bool]:
		def workhorse(node: ast.AST) -> bool: return ast.unparse(node) == ast.unparse(astAST)
		return workhorse
