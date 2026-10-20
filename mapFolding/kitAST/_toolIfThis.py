"""I use this module to extend state-pattern predicates for source generation.

(AI generated docstring)

This module defines comparison and control-flow predicates on Python syntax [1].
The exported class extends `astToolkit.IfThis` [2] with state-attribute patterns.

Contents
--------
Classes
	IfThis
		Build predicates for state comparisons and control-flow syntax.

References
----------
[1] Python abstract syntax tree node definitions.
	https://docs.python.org/3/library/ast.html
[2] astToolkit tree visitors and source containers.
	https://context7.com/hunterhogan/asttoolkit
"""
from __future__ import annotations

from astToolkit import Be, IfThis as astToolkit_IfThis
from typing import TYPE_CHECKING, TypeIs

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction
	import ast

class IfThis(astToolkit_IfThis):
	"""Build predicates for state comparisons and control-flow syntax.

	(AI generated docstring)

	You can use these static methods to recognize comparisons involving a named state attribute,
	or conditional and loop statements using those comparisons. The class extends the predicates
	in `astToolkit.IfThis` [1] with patterns used by the module generators.

	References
	----------
	[1] astToolkit tree visitors and source containers.
		https://context7.com/hunterhogan/asttoolkit
	"""

	@staticmethod
	def is0LessThanAttributeNamespaceIdentifier(namespace: str, identifier: str) -> CallableFunction[[ast.AST], TypeIs[ast.Compare]]:
		"""Build a predicate for `0 < namespace.identifier` syntax.

		(AI generated docstring)

		You can use this method to recognize an `ast.Compare` node [1] with the literal
		comparison orientation shown in the summary.

		Comparison Selection
		--------------------
		The predicate checks the first operator and first comparator of the comparison.

		Parameters
		----------
		namespace : str
			Plain object name on the left of the attribute access.
		identifier : str
			Attribute identifier to match on the named object.

		Returns
		-------
		predicate : CallableFunction[[ast.AST], TypeIs[ast.Compare]]
			Callable returning `True` when the inspected node matches the required syntax.

		Examples
		--------
		The module `mapFolding.kitAST._toolIfThis` uses this call [2].

			```python
			IfThis.is0LessThanAttributeNamespaceIdentifier(namespace, identifier)
			```

		References
		----------
		[1] Python abstract syntax tree node definitions.
			https://docs.python.org/3/library/ast.html
		[2] `mapFolding.kitAST._toolIfThis`
		"""
		return IfThis.isAllOf(Be.Compare.leftIs(IfThis.isConstant_value(0))
					, Be.Compare.opsIs(Be.at(0, Be.Lt))
					, Be.Compare.comparatorsIs(Be.at(0, IfThis.isAttributeNamespaceIdentifier(namespace, identifier)))
				)  # pyright: ignore[reportReturnType] # ty: ignore[invalid-return-type]

	@staticmethod
	def isAttributeNamespaceIdentifierLessThanOrEqual0(namespace: str, identifier: str) -> CallableFunction[[ast.AST], TypeIs[ast.Compare]]:
		"""Build a predicate for an attribute comparison beginning with `<=`.

		(AI generated docstring)

		You can use this method to recognize an `ast.Compare` node [1] whose left operand is the named
		attribute and whose first operator is `<=`.

		Parameters
		----------
		namespace : str
			Plain object name on the left of the attribute access.
		identifier : str
			Attribute identifier to match on the named object.

		Returns
		-------
		predicate : CallableFunction[[ast.AST], TypeIs[ast.Compare]]
			Callable returning `True` for the required left operand and first operator.

		References
		----------
		[1] Python abstract syntax tree node definitions.
			https://docs.python.org/3/library/ast.html
		"""
		return IfThis.isAllOf(Be.Compare.leftIs(IfThis.isAttributeNamespaceIdentifier(namespace, identifier))
					, Be.Compare.opsIs(Be.at(0, Be.LtE))
				)  # pyright: ignore[reportReturnType] # ty: ignore[invalid-return-type]

	@staticmethod
	def isAttributeNamespaceIdentifierGreaterThan0(namespace: str, identifier: str) -> CallableFunction[[ast.AST], TypeIs[ast.Compare]]:
		"""Build a predicate for `namespace.identifier > 0` syntax.

		(AI generated docstring)

		You can use this method to recognize an `ast.Compare` node [1] with the literal
		comparison orientation shown in the summary.

		Comparison Selection
		--------------------
		The predicate checks the first operator and first comparator of the comparison.

		Parameters
		----------
		namespace : str
			Plain object name on the left of the attribute access.
		identifier : str
			Attribute identifier to match on the named object.

		Returns
		-------
		predicate : CallableFunction[[ast.AST], TypeIs[ast.Compare]]
			Callable returning `True` when the inspected node matches the required syntax.

		Examples
		--------
		The module `mapFolding.kitAST._toolIfThis` uses this call [2].

			```python
			IfThis.isAttributeNamespaceIdentifierGreaterThan0(namespace, identifier)
			```

		References
		----------
		[1] Python abstract syntax tree node definitions.
			https://docs.python.org/3/library/ast.html
		[2] `mapFolding.kitAST._toolIfThis`
		"""
		return IfThis.isAllOf(Be.Compare.leftIs(IfThis.isAttributeNamespaceIdentifier(namespace, identifier))
					, Be.Compare.opsIs(Be.at(0, Be.Gt))
					, Be.Compare.comparatorsIs(Be.at(0, IfThis.isConstant_value(0)))
				)  # pyright: ignore[reportReturnType] # ty: ignore[invalid-return-type]

	@staticmethod
	def isIf0LessThanAttributeNamespaceIdentifier(namespace: str, identifier: str) -> CallableFunction[[ast.AST], TypeIs[ast.If]]:
		"""Build a predicate for `if 0 < namespace.identifier` syntax.

		(AI generated docstring)

		You can use this method to recognize an `ast.If` node [1] with the literal
		comparison orientation shown in the summary.

		Comparison Selection
		--------------------
		The predicate checks the first operator and first comparator of the comparison.

		Parameters
		----------
		namespace : str
			Plain object name on the left of the attribute access.
		identifier : str
			Attribute identifier to match on the named object.

		Returns
		-------
		predicate : CallableFunction[[ast.AST], TypeIs[ast.If]]
			Callable returning `True` when the inspected node matches the required syntax.

		References
		----------
		[1] Python abstract syntax tree node definitions.
			https://docs.python.org/3/library/ast.html
		"""
		return Be.If.testIs(IfThis.is0LessThanAttributeNamespaceIdentifier(namespace, identifier))  # ty: ignore[invalid-return-type]

	@staticmethod
	def isIfAttributeNamespaceIdentifierGreaterThan0(namespace: str, identifier: str) -> CallableFunction[[ast.AST], TypeIs[ast.If]]:
		"""Build a predicate for `if namespace.identifier > 0` syntax.

		(AI generated docstring)

		You can use this method to recognize an `ast.If` node [1] with the literal
		comparison orientation shown in the summary.

		Comparison Selection
		--------------------
		The predicate checks the first operator and first comparator of the comparison.

		Parameters
		----------
		namespace : str
			Plain object name on the left of the attribute access.
		identifier : str
			Attribute identifier to match on the named object.

		Returns
		-------
		predicate : CallableFunction[[ast.AST], TypeIs[ast.If]]
			Callable returning `True` when the inspected node matches the required syntax.

		Examples
		--------
		The module `mapFolding.kitAST.mapFolding._count` uses this call [2].

			```python
			IfThis.isIfAttributeNamespaceIdentifierGreaterThan0(名dataclassInstance, 'leaf1ndex')
			```

		References
		----------
		[1] Python abstract syntax tree node definitions.
			https://docs.python.org/3/library/ast.html
		[2] `mapFolding.kitAST.mapFolding._count`
		"""
		return Be.If.testIs(IfThis.isAttributeNamespaceIdentifierGreaterThan0(namespace, identifier))  # ty: ignore[invalid-return-type]

	@staticmethod
	def isWhile0LessThanAttributeNamespaceIdentifier(namespace: str, identifier: str) -> CallableFunction[[ast.AST], TypeIs[ast.While]]:
		"""Build a predicate for `while 0 < namespace.identifier` syntax.

		(AI generated docstring)

		You can use this method to recognize an `ast.While` node [1] with the literal
		comparison orientation shown in the summary.

		Comparison Selection
		--------------------
		The predicate checks the first operator and first comparator of the comparison.

		Parameters
		----------
		namespace : str
			Plain object name on the left of the attribute access.
		identifier : str
			Attribute identifier to match on the named object.

		Returns
		-------
		predicate : CallableFunction[[ast.AST], TypeIs[ast.While]]
			Callable returning `True` when the inspected node matches the required syntax.

		Examples
		--------
		The module `mapFolding.kitAST.matrixMeanders.makeModules` uses this call [2].

			```python
			IfThis.isWhile0LessThanAttributeNamespaceIdentifier(名DataclassInstance, 'boundary')
			```

		References
		----------
		[1] Python abstract syntax tree node definitions.
			https://docs.python.org/3/library/ast.html
		[2] `mapFolding.kitAST.matrixMeanders.makeModules`
		"""
		return Be.While.testIs(IfThis.is0LessThanAttributeNamespaceIdentifier(namespace, identifier))  # ty: ignore[invalid-return-type]

	@staticmethod
	def isWhileAttributeNamespaceIdentifierGreaterThan0(namespace: str, identifier: str) -> CallableFunction[[ast.AST], TypeIs[ast.While]]:
		"""Build a predicate for `while namespace.identifier > 0` syntax.

		(AI generated docstring)

		You can use this method to recognize an `ast.While` node [1] with the literal
		comparison orientation shown in the summary.

		Comparison Selection
		--------------------
		The predicate checks the first operator and first comparator of the comparison.

		Parameters
		----------
		namespace : str
			Plain object name on the left of the attribute access.
		identifier : str
			Attribute identifier to match on the named object.

		Returns
		-------
		predicate : CallableFunction[[ast.AST], TypeIs[ast.While]]
			Callable returning `True` when the inspected node matches the required syntax.

		Examples
		--------
		The module `mapFolding.kitAST.mapFolding._count` uses this call [2].

			```python
			IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(名dataclassInstance, 'leaf1ndex')
			```

		References
		----------
		[1] Python abstract syntax tree node definitions.
			https://docs.python.org/3/library/ast.html
		[2] `mapFolding.kitAST.mapFolding._count`
		"""
		return Be.While.testIs(IfThis.isAttributeNamespaceIdentifierGreaterThan0(namespace, identifier))  # ty: ignore[invalid-return-type]
