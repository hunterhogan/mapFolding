"""
AST Node Transformation Actions for Python Code Manipulation

This module provides the Then class with static methods for generating callable
action functions that specify what to do with AST nodes that match predicates.
These action functions are used primarily with NodeChanger and NodeTourist to
transform or extract information from AST nodes.

The module also contains the grab class that provides functions for modifying
specific attributes of AST nodes while preserving their structure, enabling
fine-grained control when transforming AST structures.

Together, these classes provide a complete system for manipulating AST nodes
once they have been identified using predicate functions from ifThis.
"""

from collections.abc import Callable, Sequence
from mapFolding.someAssemblyRequired import ast_Identifier, astClassHasDOTvalue
from typing import Any
import ast

class grab:
	"""
	Modify specific attributes of AST nodes while preserving the node structure.

	The grab class provides static methods that create transformation functions to modify
	specific attributes of AST nodes. Unlike DOT which provides read-only access,
	grab allows for targeted modifications of node attributes without replacing the
	entire node.

	Each method returns a function that takes a node, applies a transformation to a
	specific attribute of that node, and returns the modified node. This enables fine-grained
	control when transforming AST structures.
	"""
	@staticmethod
	def argAttribute(action: Callable[[ast_Identifier | None], ast_Identifier]) -> Callable[[ast.arg | ast.keyword], ast.arg | ast.keyword]:
		def workhorse(node: ast.arg | ast.keyword) -> ast.arg | ast.keyword:
			node.arg = action(node.arg)
			return node
		return workhorse

	@staticmethod
	def funcAttribute(action: Callable[[ast.expr], ast.expr]) -> Callable[[ast.Call], ast.Call]:
		def workhorse(node: ast.Call) -> ast.Call:
			node.func = action(node.func)
			return node
		return workhorse

	@staticmethod
	def idAttribute(action: Callable[[ast_Identifier], ast_Identifier]) -> Callable[[ast.Name], ast.Name]:
		def workhorse(node: ast.Name) -> ast.Name:
			node.id = action(node.id)
			return node
		return workhorse

	@staticmethod
	def valueAttribute(action: Callable[[Any | ast.expr | bool | None], Any]) -> Callable[[astClassHasDOTvalue], astClassHasDOTvalue]:
		def workhorse(node: astClassHasDOTvalue) -> astClassHasDOTvalue:
			node.value = action(node.value)
			return node
		return workhorse

class Then:
	"""
	Provide action functions that specify what to do with AST nodes that match predicates.

	The Then class contains static methods that generate action functions used with
	NodeChanger and NodeTourist to transform or extract information from AST nodes
	that match specific predicates. These actions include node replacement, insertion,
	extraction, and collection operations.

	When paired with predicates from the ifThis class, Then methods complete the
	pattern-matching-and-action workflow for AST manipulation.
	"""
	@staticmethod
	def appendTo(listOfAny: list[Any]) -> Callable[[ast.AST | ast_Identifier], list[Any]]:
		def workhorse(node: ast.AST | ast_Identifier) -> list[Any]:
			listOfAny.append(node)
			return listOfAny
		return workhorse

	@staticmethod
	def extractIt(node: ast.AST) -> ast.AST | ast_Identifier:
		return node

	@staticmethod
	def insertThisAbove(list_astAST: Sequence[ast.AST]) -> Callable[[ast.AST], Sequence[ast.AST]]:
		return lambda aboveMe: [*list_astAST, aboveMe]

	@staticmethod
	def insertThisBelow(list_astAST: Sequence[ast.AST]) -> Callable[[ast.AST], Sequence[ast.AST]]:
		return lambda belowMe: [belowMe, *list_astAST]

	@staticmethod
	def removeIt(_node: ast.AST) -> None: return None

	@staticmethod
	def replaceWith(astAST: Any) -> Callable[[Any], Any]:
		return lambda _replaceMe: astAST

	@staticmethod
	def updateKeyValueIn(key: Callable[..., Any], value: Callable[..., Any], dictionary: dict[Any, Any]) -> Callable[[ast.AST], dict[Any, Any]]:
		def workhorse(node: ast.AST) -> dict[Any, Any]:
			dictionary.setdefault(key(node), value(node))
			return dictionary
		return workhorse
