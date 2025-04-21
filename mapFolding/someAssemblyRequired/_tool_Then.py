"""
AST Node Transformation Actions for Python Code Manipulation

This module provides the Then class with static methods for generating callable action functions that specify what to do
with AST nodes that match predicates. These action functions are used primarily with NodeChanger and NodeTourist to
transform or extract information from AST nodes.

The module also contains the grab class that provides functions for modifying specific attributes of AST nodes while
preserving their structure, enabling fine-grained control when transforming AST structures.

Together, these classes provide a complete system for manipulating AST nodes once they have been identified using
predicate functions from ifThis.
"""

from collections.abc import Callable, Sequence
from mapFolding.someAssemblyRequired import ast_Identifier, astClassHasDOTvalue, ImaCallToName, NodeORattribute
from typing import Any
import ast

class grab:
	"""
	Modify specific attributes of AST nodes while preserving the node structure.

	The grab class provides static methods that create transformation functions to modify specific attributes of AST
	nodes. Unlike DOT which provides read-only access, grab allows for targeted modifications of node attributes without
	replacing the entire node.

	Each method returns a function that takes a node, applies a transformation to a specific attribute of that node, and
	returns the modified node. This enables fine-grained control when transforming AST structures.
	"""
	@staticmethod
	def andDoAllOf(listOfActions: list[Callable[[NodeORattribute], NodeORattribute]]) -> Callable[[NodeORattribute], NodeORattribute]:
		def workhorse(node: NodeORattribute) -> NodeORattribute:
			for action in listOfActions:
				node = action(node)
			return node
		return workhorse

	@staticmethod
	def argAttribute(action: Callable[[ast_Identifier | None], ast_Identifier]) -> Callable[[ast.arg | ast.keyword], ast.arg | ast.keyword]:
		def workhorse(node: ast.arg | ast.keyword) -> ast.arg | ast.keyword:
			node.arg = action(node.arg)
			return node
		return workhorse

	@staticmethod
	def attrAttribute(action: Callable[[ast_Identifier], ast_Identifier]) -> Callable[[ast.Attribute], ast.Attribute]:
		def workhorse(node: ast.Attribute) -> ast.Attribute:
			node.attr = action(node.attr)
			return node
		return workhorse

	@staticmethod
	def comparatorsAttribute(action: Callable[[list[ast.expr]], list[ast.expr]]) -> Callable[[ast.Compare], ast.Compare]:
		def workhorse(node: ast.Compare) -> ast.Compare:
			node.comparators = action(node.comparators)
			return node
		return workhorse

	@staticmethod
	def funcAttribute(action: Callable[[ast.expr], ast.expr]) -> Callable[[ast.Call], ast.Call]:
		def workhorse(node: ast.Call) -> ast.Call:
			node.func = action(node.func)
			return node
		return workhorse

	@staticmethod
	def funcDOTidAttribute(action: Callable[[ast_Identifier], Any]) -> Callable[[ImaCallToName], ImaCallToName]:
		def workhorse(node: ImaCallToName) -> ImaCallToName:
			node.func = grab.idAttribute(action)(node.func)
			return node
		return workhorse

	@staticmethod
	def idAttribute(action: Callable[[ast_Identifier], ast_Identifier]) -> Callable[[ast.Name], ast.Name]:
		def workhorse(node: ast.Name) -> ast.Name:
			node.id = action(node.id)
			return node
		return workhorse

	@staticmethod
	def leftAttribute(action: Callable[[ast.expr], ast.expr]) -> Callable[[ast.BinOp | ast.Compare], ast.BinOp | ast.Compare]:
		def workhorse(node: ast.BinOp | ast.Compare) -> ast.BinOp | ast.Compare:
			node.left = action(node.left)
			return node
		return workhorse

	@staticmethod
	def opsAttribute(action: Callable[[list[ast.cmpop]], list[ast.cmpop]]) -> Callable[[ast.Compare], ast.Compare]:
		def workhorse(node: ast.Compare) -> ast.Compare:
			node.ops = action(node.ops)
			return node
		return workhorse

	@staticmethod
	def testAttribute(action: Callable[[ast.expr], ast.expr]) -> Callable[[ast.Assert | ast.If | ast.IfExp | ast.While], ast.Assert | ast.If | ast.IfExp | ast.While]:
		def workhorse(node: ast.Assert | ast.If | ast.IfExp | ast.While) -> ast.Assert | ast.If | ast.IfExp | ast.While:
			node.test = action(node.test)
			return node
		return workhorse

	@staticmethod
	def valueAttribute(action: Callable[[Any], Any]) -> Callable[[astClassHasDOTvalue], astClassHasDOTvalue]:
		def workhorse(node: astClassHasDOTvalue) -> astClassHasDOTvalue:
			node.value = action(node.value)
			return node
		return workhorse

class Then:
	"""
	Provide action functions that specify what to do with AST nodes that match predicates.

	The Then class contains static methods that generate action functions used with NodeChanger and NodeTourist to
	transform or extract information from AST nodes that match specific predicates. These actions include node
	replacement, insertion, extraction, and collection operations.

	When paired with predicates from the ifThis class, Then methods complete the pattern-matching-and-action workflow
	for AST manipulation.
	"""
	@staticmethod
	def appendTo(listOfAny: list[Any]) -> Callable[[ast.AST | ast_Identifier], ast.AST | ast_Identifier]:
		def workhorse(node: ast.AST | ast_Identifier) -> ast.AST | ast_Identifier:
			listOfAny.append(node)
			return node
		return workhorse

	@staticmethod
	def extractIt(node: NodeORattribute) -> NodeORattribute:
		return node

	@staticmethod
	def insertThisAbove(list_astAST: Sequence[ast.AST]) -> Callable[[ast.AST], Sequence[ast.AST]]:
		return lambda aboveMe: [*list_astAST, aboveMe]

	@staticmethod
	def insertThisBelow(list_astAST: Sequence[ast.AST]) -> Callable[[ast.AST], Sequence[ast.AST]]:
		return lambda belowMe: [belowMe, *list_astAST]

	@staticmethod
	def removeIt(_removeMe: ast.AST) -> None:
		return None

	@staticmethod
	def replaceWith(astAST: NodeORattribute) -> Callable[[NodeORattribute], NodeORattribute]:
		return lambda _replaceMe: astAST

	@staticmethod
	def updateKeyValueIn(key: Callable[..., Any], value: Callable[..., Any], dictionary: dict[Any, Any]) -> Callable[[ast.AST], dict[Any, Any]]:
		def workhorse(node: ast.AST) -> dict[Any, Any]:
			dictionary.setdefault(key(node), value(node))
			return dictionary
		return workhorse
