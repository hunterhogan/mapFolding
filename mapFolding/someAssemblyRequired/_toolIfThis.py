"""
Map folding AST transformation system: Foundation pattern matching and filtering predicates.

This module establishes the foundational layer of the map folding AST transformation system by
providing specialized predicate functions for identifying specific code patterns in Abstract
Syntax Trees. The transformation system converts high-level dataclass-based map folding algorithms
into optimized computational modules through systematic AST manipulation, beginning with precise
pattern recognition capabilities defined here.

The predicates extend `astToolkit.IfThis` with domain-specific functions that recognize conditional
expressions and control flow structures essential to map folding computations. These patterns include
attribute comparisons against specific values, loop termination conditions based on counting variables,
and bounds checking operations that characterize map folding algorithm structures.

Pattern recognition serves as the entry point for the broader transformation system that decomposes
dataclass-dependent algorithms, applies performance optimizations, and generates specialized modules
optimized for Numba just-in-time compilation. The precise identification of these structural patterns
enables subsequent transformation stages to apply targeted optimizations while preserving mathematical
correctness.

Classes:
    IfThis: Extended predicate class with specialized methods for matching attribute comparisons
            and control flow patterns involving namespaced identifiers essential to map folding
            algorithm transformations.
"""

from astToolkit import Be, IfThis as astToolkit_IfThis
from collections.abc import Callable
from typing_extensions import TypeIs
import ast

class IfThis(astToolkit_IfThis):
	"""Provide predicate functions for matching and filtering AST nodes based on various criteria.

	(AI generated docstring)

	The `IfThis` `class` contains static methods that generate predicate functions used to test whether AST nodes match
	specific criteria. These predicates can be used with `NodeChanger` and `NodeTourist` to identify and process specific
	patterns in the AST.

	The `class` provides predicates for matching various node types, attributes, identifiers, and structural patterns,
	enabling precise targeting of AST elements for analysis or transformation.

	"""

	@staticmethod
	def isAttributeNamespaceIdentifierLessThanOrEqual0(namespace: str, identifier: str) -> Callable[[ast.AST], TypeIs[ast.Compare]]:
		"""Generate a predicate that matches comparison expressions testing if a namespaced attribute is less than or equal to 0.

		(AI generated docstring)

		This function creates a predicate that identifies AST nodes representing comparisons
		of the form `namespace.identifier <= 0`. It's used to identify conditional
		expressions that test non-positive values of counting variables or similar constructs.

		Parameters
		----------
		namespace : str
			The namespace or object name containing the attribute.
		identifier : str
			The attribute name to test.

		Returns
		-------
		predicate : Callable[[ast.AST], TypeIs[ast.Compare]]
			A predicate function that returns `True` for `Compare` nodes matching the pattern.

		"""
		return lambda node: (Be.Compare.leftIs(IfThis.isAttributeNamespaceIdentifier(namespace, identifier))(node)
					and Be.Compare.opsIs(lambda at: Be.LtE(at[0]))(node)
				)

	@staticmethod
	def isAttributeNamespaceIdentifierGreaterThan0(namespace: str, identifier: str) -> Callable[[ast.AST], TypeIs[ast.Compare] | bool]:
		"""Generate a predicate that matches comparison expressions testing if a namespaced attribute is greater than 0.

		(AI generated docstring)

		This function creates a predicate that identifies AST nodes representing comparisons
		of the form `namespace.identifier > 0`. It's commonly used to identify conditional
		expressions that test positive values of counting variables or similar constructs.

		Parameters
		----------
		namespace : str
			The namespace or object name containing the attribute.
		identifier : str
			The attribute name to test.

		Returns
		-------
		predicate : Callable[[ast.AST], TypeIs[ast.Compare]]
			A predicate function that returns `True` for `Compare` nodes matching the pattern.

		"""
		return lambda node: (Be.Compare.leftIs(IfThis.isAttributeNamespaceIdentifier(namespace, identifier))(node)
					and Be.Gt(node.ops[0])
					and IfThis.isConstant_value(0)(node.comparators[0]))

	@staticmethod
	def isIfAttributeNamespaceIdentifierGreaterThan0(namespace: str, identifier: str) -> Callable[[ast.AST], TypeIs[ast.If]]:
		"""Generate a predicate that matches If statements testing if a namespaced attribute is greater than 0.

		(AI generated docstring)

		This function creates a predicate that identifies AST nodes representing conditional
		statements of the form `if namespace.identifier > 0:`. It's used to find control
		flow structures that depend on positive values of specific attributes.

		Parameters
		----------
		namespace : str
			The namespace or object name containing the attribute.
		identifier : str
			The attribute name to test.

		Returns
		-------
		predicate : Callable[[ast.AST], TypeIs[ast.If]]
			A predicate function that returns `True` for `If` nodes with the specified test condition.

		"""
		return Be.If.testIs(IfThis.isAttributeNamespaceIdentifierGreaterThan0(namespace, identifier))

	@staticmethod
	def isWhileAttributeNamespaceIdentifierGreaterThan0(namespace: str, identifier: str) -> Callable[[ast.AST], TypeIs[ast.While]]:
		"""Generate a predicate that matches While loops testing if a namespaced attribute is greater than 0.

		(AI generated docstring)

		This function creates a predicate that identifies AST nodes representing loop
		statements of the form `while namespace.identifier > 0:`. It's used to find
		iteration constructs that continue while specific attributes remain positive.

		Parameters
		----------
		namespace : str
			The namespace or object name containing the attribute.
		identifier : str
			The attribute name to test.

		Returns
		-------
		predicate : Callable[[ast.AST], TypeIs[ast.While]]
			A predicate function that returns `True` for `While` nodes with the specified test condition.

		"""
		return Be.While.testIs(IfThis.isAttributeNamespaceIdentifierGreaterThan0(namespace, identifier))
