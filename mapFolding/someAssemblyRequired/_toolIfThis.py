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

from astToolkit import Be, DOT, IfThis as astToolkit_IfThis
from collections.abc import Callable
from typing import TypeGuard
import ast

class IfThis(astToolkit_IfThis):
	"""
	Provide predicate functions for matching and filtering AST nodes based on various criteria.

	The ifThis class contains static methods that generate predicate functions used to test whether AST nodes match
	specific criteria. These predicates can be used with NodeChanger and NodeTourist to identify and process specific
	patterns in the AST.

	The class provides predicates for matching various node types, attributes, identifiers, and structural patterns,
	enabling precise targeting of AST elements for analysis or transformation.
	"""
	@staticmethod
	def isAttributeNamespaceIdentifierGreaterThan0(namespace: str, identifier: str) -> Callable[[ast.AST], TypeGuard[ast.Compare] | bool]:
		return lambda node: (Be.Compare(node)
					and IfThis.isAttributeNamespace_Identifier(namespace, identifier)(DOT.left(node))
					and Be.Gt(node.ops[0])
					and IfThis.isConstant_value(0)(node.comparators[0]))
	@staticmethod
	def isIfAttributeNamespaceIdentifierGreaterThan0(namespace: str, identifier: str) -> Callable[[ast.AST], TypeGuard[ast.If] | bool]:
		return lambda node: (Be.If(node)
					and IfThis.isAttributeNamespaceIdentifierGreaterThan0(namespace, identifier)(DOT.test(node)))

	@staticmethod
	def isWhileAttributeNamespaceIdentifierGreaterThan0(namespace: str, identifier: str) -> Callable[[ast.AST], TypeGuard[ast.While] | bool]:
		return lambda node: (Be.While(node)
					and IfThis.isAttributeNamespaceIdentifierGreaterThan0(namespace, identifier)(DOT.test(node)))

	@staticmethod
	def isAttributeNamespaceIdentifierLessThanOrEqual0(namespace: str, identifier: str) -> Callable[[ast.AST], TypeGuard[ast.Compare] | bool]:
		return lambda node: (Be.Compare(node)
					and IfThis.isAttributeNamespace_Identifier(namespace, identifier)(DOT.left(node))
					and Be.LtE(node.ops[0]))
