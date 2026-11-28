"""Generate tests of OEIS formulas and triangle relations, and their reference samples.

(AI generated docstring)

The parameters come from the same symbolic relations that generate `mapFolding.oeis.byFormula`.
Every expected value of a generated test reads `lookupValues` in
`mapFolding.tests.dataSamples.oeis`, a static copy of the stored OEIS b-file values.

Contents
--------
Functions
	callIndexInteger吗
		Check whether a symbolic sequence call has an integer first argument.
	callTriangle吗
		Check whether a symbolic call names a registered triangle.
	formulaApplies吗
		Check whether a formula's condition holds at an index.
	formulaBounded吗
		Check a formula test and its dependencies against the test selection limits.
	formulaTriangle吗
		Check whether a symbolic expression contains a registered triangle.
	getFormula
		Obtain the expression from a symbolic solution.
	getReference
		Obtain the sequence ID and integer index from a symbolic call.
	getReferenceFormula
		Obtain the sequence ID and index from a formula test parameter.
	getValueKnown
		Obtain a stored sequence value at a specified index.
	identifyRelations
		Give each relation of the same sequences a numbered identifier.
	instantiateRelation
		Instantiate a relation where every required sequence value is stored.
	instantiateSide
		Evaluate one side of a relation and flatten its triangle indices.
	makeFormulaParameters
		Select the formula selectors and indices whose evaluation the test settings bound.
	makeIdentifier_f
		Make the formula selector's contribution to a pytest parameter ID.
	makeParameterFormula
		Make a formula test parameter from a symbolic sequence call.
	makeParametersFormulaSequence
		Collect candidate formula tests for one sequence.
	makeParametersFormulaTriangle
		Collect candidate tests for a formula that reads a triangle.
	makeReferences
		Collect the sequence values that the generated tests read.
	makeReferencesRelation
		Collect the sequence references from both sides of a relation test.
	makeRelationParameters
		Instantiate each triangle relation at every index whose sequence values are stored.
	makeSamplesOEIS
		Make the samples module without reference values.
	makeTestsByFormula
		Make the module that tests each function in `mapFolding.oeis.byFormula`.
	makeTestsTriangle
		Make the module that tests each triangle relation.
	make_astLookupValues
		Represent a stored reference value as an AST lookup.
	make_astModuleTests
		Assemble a documented test module from a function and its imports.
	make_astParameterFormula
		Represent a formula test parameter and its reference value as a pytest call.
	make_astParameterRelation
		Represent an instantiated relation test as a pytest parameter call.
	make_astReference
		Represent a sequence reference as an AST tuple.
	relationTriangle吗
		Check whether a symbolic relation is an equality containing a registered triangle.
	selectFormula
		Select the first applicable formula and substitute its index.
	sumBounded吗
		Check the length of a sum's first index range against the configured limit.
	updateSamplesOEIS
		Add the missing reference values to the samples module.

Test settings
	boxOfIndicesLarge
		Additional sequence indices considered alongside the initial indices.
	countCallsMaximum
		Maximum number of formula requests examined for one candidate test.
	countIndicesFirst
		Number of candidate indices beginning at each sequence's offset.
	lengthSumMaximum
		Maximum accepted length of a sum's first index range.
	nCountedMaximum
		Maximum fallback index for counting sequences, apart from their offset.
	nDerivedMaximum
		Maximum fallback index for other sequences, apart from their offset.
"""

from __future__ import annotations

from ast import literal_eval
from astToolkit import Be, Grab, IfThis, Make, NodeChanger, NodeTourist, Then
from collections import deque
from functools import partial
from humpy_cytoolz import compose_left, first, groupby as toolz_groupby, unique
from hunterMakesPy import raiseIfNone
from itertools import chain, compress, count, filterfalse, product as CartesianProduct, repeat, starmap
from mapFolding.kitAST import IngredientsFunction, IngredientsModule
from mapFolding.kitAST.oeis.oeisIDByFormula import make_ast_exprSymbolic, make_f, makeBoxOf_f
from mapFolding.oeis import boxOfOEISidTriangle, getMetadata, getValuesKnown, settingsOEIS
from mapFolding.oeis._symbolic import callStored吗, getOEISid, makeEquationFlat, makeIndicesCall, makeIndicesRelation, makeTermLast
from mapFolding.oeis._symbolicRelations import boxOfRelations
from operator import eq, itemgetter
from sympy import Equality, Expr, Integer, oo, Range, Sum, true
from sympy.core.function import AppliedUndef
from typing import TYPE_CHECKING
import re as regex

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator
	from mapFolding.oeis import OEISid
	from mapFolding.oeis._symbolicRelations import 形Relation
	from sympy import Basic
	from sympy.logic.boolalg import Boolean
	import ast

countIndicesFirst: int = 8
boxOfIndicesLarge: tuple[int, ...] = (31, 129, 300)
countCallsMaximum: int = 256
lengthSumMaximum: int = 32
nCountedMaximum: int = 7
nDerivedMaximum: int = 4

def getFormula(solution: tuple[OEISid, Expr, Boolean]) -> Expr:
	"""Obtain the formula expression from a symbolic solution.

	(AI generated docstring)

	Parameters
	----------
	solution : tuple[OEISid, Expr, Boolean]
		Sequence ID, formula, and condition under which the formula applies.

	Returns
	-------
	formula : Expr
		The supplied formula, with no substitutions or evaluation.
	"""
	_oeisID, formula, _condition = solution
	return formula

def formulaApplies吗(n: int, solution: tuple[OEISid, Expr, Boolean]) -> bool:
	"""Check whether a formula's applicability condition holds at `n`.

	(AI generated docstring)

	Parameters
	----------
	n : int
		Sequence index to substitute for `settingsOEIS.n` in the condition.
	solution : tuple[OEISid, Expr, Boolean]
		Sequence ID, formula, and applicability condition.

	Returns
	-------
	applies : bool
		Whether the substituted condition equals SymPy's `true`.
	"""
	_oeisID, _formula, condition = solution
	return condition.subs(settingsOEIS.n, n) == true

def formulaTriangle吗(formula: Basic) -> bool:
	"""Check whether a symbolic expression contains a registered triangle function.

	(AI generated docstring)

	Parameters
	----------
	formula : Basic
		Symbolic expression or relation to inspect.

	Returns
	-------
	containsTriangle : bool
		Whether any function registered in `boxOfOEISidTriangle` occurs in `formula`.
	"""
	return formula.has(*map(settingsOEIS.lookupSymbolic.__getitem__, boxOfOEISidTriangle))

def callTriangle吗(call: AppliedUndef) -> bool:
	"""Check whether a symbolic call names a registered OEIS triangle.

	(AI generated docstring)

	Parameters
	----------
	call : AppliedUndef
		Call to a sequence function registered in `settingsOEIS.lookupID`.

	Returns
	-------
	triangle : bool
		Whether the sequence ID belongs to `boxOfOEISidTriangle`, regardless of the arguments.
	"""
	return getOEISid(call) in boxOfOEISidTriangle

def callIndexInteger吗(call: AppliedUndef) -> bool:
	"""Check whether a symbolic sequence call has an integer first argument.

	(AI generated docstring)

	Parameters
	----------
	call : AppliedUndef
		Symbolic call with at least one argument.

	Returns
	-------
	indexInteger : bool
		Whether the first argument is a SymPy `Integer`. Other arguments are not checked.
	"""
	return isinstance(call.args[0], Integer)

def sumBounded吗(summation: Sum) -> bool:
	"""Check whether a sum's first index range satisfies `lengthSumMaximum`.

	(AI generated docstring)

	Parameters
	----------
	summation : Sum
		Symbolic sum whose first limit supplies an index, a lower bound, and an upper bound.

	Returns
	-------
	bounded : bool
		Whether upper minus lower plus one is an integer no greater than `lengthSumMaximum`.
		Additional index ranges and the summand's cost are not checked.
	"""
	_k, lower, upper = summation.limits[0]
	length = upper - lower + 1
	return length.is_Integer is True and bool(length <= lengthSumMaximum)

def selectFormula(n: int, f: str | None, boxOfSolutions: list[tuple[OEISid, Expr, Boolean]]) -> Expr | None:
	"""Select the first applicable formula and substitute the sequence index.

	(AI generated docstring)

	Parameters
	----------
	n : int
		Sequence index at which the formula must apply.
	f : str | None
		Formula selector assigned by `makeBoxOf_f`. Matching formulas take precedence over
		formulas with an empty selector. `None` considers only formulas with an empty selector.
	boxOfSolutions : list[tuple[OEISid, Expr, Boolean]]
		Solutions for one sequence, in precedence order, each containing the sequence ID,
		formula, and applicability condition.

	Returns
	-------
	expression : Expr | None
		First applicable formula with `settingsOEIS.n` replaced by `n`, or `None` when no
		candidate applies. Sequence calls and sums remain symbolic.
	"""
	boxOf_f: tuple[str, ...] = makeBoxOf_f(boxOfSolutions)
	boxOfCandidates: Iterator[tuple[OEISid, Expr, Boolean]] = chain(compress(boxOfSolutions, map(partial(eq, f), boxOf_f))
		, compress(boxOfSolutions, map(partial(eq, ''), boxOf_f)))
	solution: tuple[OEISid, Expr, Boolean] | None = next(filter(partial(formulaApplies吗, n), boxOfCandidates), None)
	expression: Expr | None = None
	if solution is not None:
		_oeisID, formula, _condition = solution
		expression = formula.subs(settingsOEIS.n, n)
	return expression

def formulaBounded吗(lookupSolutions: dict[OEISid, list[tuple[OEISid, Expr, Boolean]]], parameter: tuple[OEISid, int, str | None]) -> bool:
	"""Check a formula test and its dependencies against the test selection limits.

	(AI generated docstring)

	Every requested index must be at least its sequence's offset. A request without an applicable
	formula must satisfy `nCountedMaximum` for counting sequences or `nDerivedMaximum` for
	other sequences; the sequence's offset is always allowed. Sums must satisfy
	`sumBounded吗`, and at most `countCallsMaximum` requests may be examined.

	A formula containing a triangle requires stored values for every call after sum evaluation.
	Other formulas require integer first arguments. Non-triangle dependencies are checked
	with selector `None`. These checks select test parameters without computing sequence counts.

	Parameters
	----------
	lookupSolutions : dict[OEISid, list[tuple[OEISid, Expr, Boolean]]]
		Ordered symbolic solutions for every sequence reached by the candidate test.
	parameter : tuple[OEISid, int, str | None]
		Sequence ID, index, and formula selector of the candidate test.

	Returns
	-------
	bounded : bool
		Whether every request satisfies the limits and no dependency remains unchecked.
	"""
	boxOfPending: deque[tuple[OEISid, int, str | None]] = deque((parameter,))
	countCalls: int = 0
	bounded: bool = True
	while bounded and boxOfPending and countCalls < countCallsMaximum:
		oeisID, n, f = boxOfPending.popleft()
		countCalls += 1
		bounded = getMetadata(oeisID)['offset'] <= n
		if bounded:
			expression: Expr | None = selectFormula(n, f, lookupSolutions[oeisID])
			if expression is None:
				limit: int = nDerivedMaximum
				if oeisID in settingsOEIS.boxOfIDsCounting:
					limit = nCountedMaximum
				bounded = n <= limit or n == getMetadata(oeisID)['offset']
			else:
				bounded = all(map(sumBounded吗, expression.atoms(Sum)))
				if bounded:
					boxOfCalls: set[AppliedUndef] = expression.doit().atoms(AppliedUndef)
					if formulaTriangle吗(expression):
						bounded = all(map(callStored吗, boxOfCalls))
					else:
						bounded = all(map(callIndexInteger吗, boxOfCalls))
					if bounded:
						boxOfPending.extend(map(makeParameterFormula, filterfalse(callTriangle吗, boxOfCalls)))
	return bounded and not boxOfPending

def makeParameterFormula(call: AppliedUndef) -> tuple[OEISid, int, str | None]:
	"""Make a formula test parameter for a symbolic call with the default selector.

	(AI generated docstring)

	Parameters
	----------
	call : AppliedUndef
		Registered sequence call with exactly one SymPy `Integer` argument.

	Returns
	-------
	parameter : tuple[OEISid, int, str | None]
		Sequence ID, integer index, and selector `None`.
	"""
	return (*getReference(call), None)

def makeParametersFormulaTriangle(f: str, solution: tuple[OEISid, Expr, Boolean]) -> Iterator[tuple[OEISid, int, str | None]]:
	"""Collect candidate tests for a formula that reads registered triangles.

	(AI generated docstring)

	Candidate indices come from the stored triangle rows referenced by the formula. The
	formula's final summand supplies the row index for a sum. A candidate must have a stored
	value for the result sequence and satisfy the formula's applicability condition.

	Parameters
	----------
	f : str
		Selector to attach to each candidate test.
	solution : tuple[OEISid, Expr, Boolean]
		Result sequence ID, formula containing triangle calls, and applicability condition.

	Returns
	-------
	boxOfParameters : Iterator[tuple[OEISid, int, str | None]]
		Sequence ID, index, and `f` for each candidate, ordered by increasing index. Candidates
		still require the dependency and computation checks in `formulaBounded吗`.
	"""
	oeisID, formula, _condition = solution
	boxOfIndices: frozenset[int] = frozenset[int]().union(*map(partial(makeIndicesCall, Range(getMetadata(oeisID)['offset'], oo))
		, filter(callTriangle吗, formula.replace(Sum, makeTermLast).atoms(AppliedUndef))))
	boxOfStored: Iterator[int] = filter(partial(formulaApplies吗, solution=solution), sorted(boxOfIndices.intersection(getValuesKnown(oeisID))))
	return CartesianProduct((oeisID,), boxOfStored, (f,))

def makeParametersFormulaSequence(oeisID: OEISid, boxOfSolutions: list[tuple[OEISid, Expr, Boolean]]) -> Iterator[tuple[OEISid, int, str | None]]:
	"""Collect candidate formula tests for one OEIS sequence.

	(AI generated docstring)

	Candidates use stored indices among the first `countIndicesFirst` indices from the offset
	and `boxOfIndicesLarge`, paired with each nonempty formula selector and `None`. Formulas
	that read triangles contribute the additional candidates from `makeParametersFormulaTriangle`.

	Parameters
	----------
	oeisID : OEISid
		Sequence whose formulas and stored reference values supply the candidates.
	boxOfSolutions : list[tuple[OEISid, Expr, Boolean]]
		Symbolic solutions for `oeisID`, in precedence order.

	Returns
	-------
	boxOfParameters : Iterator[tuple[OEISid, int, str | None]]
		Candidate sequence IDs, indices, and selectors. Candidates can repeat and have not
		been checked by `formulaBounded吗`.
	"""
	offset: int = getMetadata(oeisID)['offset']
	boxOfIndices: tuple[int, ...] = tuple(filter(getValuesKnown(oeisID).__contains__
		, unique((*range(offset, offset + countIndicesFirst), *boxOfIndicesLarge))))
	boxOf_f: tuple[str, ...] = makeBoxOf_f(boxOfSolutions)
	boxOfSelectors: tuple[str | None, ...] = (*filter(None, boxOf_f), None)
	boxOfSolutionsTriangle: Iterator[tuple[str, tuple[OEISid, Expr, Boolean]]] = filter(
		compose_left(itemgetter(1), getFormula, formulaTriangle吗), zip(boxOf_f, boxOfSolutions, strict=True))
	return chain(CartesianProduct((oeisID,), boxOfIndices, boxOfSelectors)
		, chain.from_iterable(starmap(makeParametersFormulaTriangle, boxOfSolutionsTriangle)))

def makeFormulaParameters(lookupSolutions: dict[OEISid, list[tuple[OEISid, Expr, Boolean]]]) -> tuple[tuple[OEISid, int, str | None], ...]:
	"""Select, for each sequence, the indices and formula selectors whose evaluation the bounds in this module limit.

	(AI generated docstring)

	Each sequence contributes its first `countIndicesFirst` indices and `boxOfIndicesLarge`, crossed with every selector and `None`,
	and each formula that reads a triangle contributes every stored index at which the formula applies.
	"""
	boxOfParameters: Iterator[tuple[OEISid, int, str | None]] = chain.from_iterable(
		starmap(makeParametersFormulaSequence, sorted(lookupSolutions.items())))
	return tuple(filter(partial(formulaBounded吗, lookupSolutions), unique(boxOfParameters)))

def makeIdentifier_f(f: str | None) -> str:
	"""Make the formula selector's contribution to a pytest parameter ID.

	(AI generated docstring)

	Parameters
	----------
	f : str | None
		Formula selector. `None` and empty text use `'default'`.

	Returns
	-------
	identifier : str
		Selector text with each run of non-word characters replaced by an underscore.
	"""
	return regex.sub(r'\W+', '_', f or 'default')

def make_astParameterFormula(parameter: tuple[OEISid, int, str | None]) -> ast.Call:
	"""Represent a formula test parameter and its stored reference as a `pytest.param` call.

	(AI generated docstring)

	Parameters
	----------
	parameter : tuple[OEISid, int, str | None]
		Sequence ID, sequence index, and formula selector.

	Returns
	-------
	astCall : ast.Call
		Call supplying the sequence ID, index, selector, and expected value from `lookupValues`.
		The parameter ID joins the sequence ID, index, and normalized selector with hyphens.
	"""
	oeisID, n, f = parameter
	return Make.Call(Make.Attribute(Make.Name('pytest'), 'param'), [*map(Make.Constant, parameter), make_astLookupValues(oeisID, Make.Constant(n))]
		, list_keyword=[Make.keyword('id', Make.Constant(f'{oeisID}-{n}-{makeIdentifier_f(f)}'))])

def relationTriangle吗(relation: 形Relation) -> bool:
	"""Check whether a symbolic relation is an equality containing a registered triangle.

	(AI generated docstring)

	Parameters
	----------
	relation : 形Relation
		Symbolic relation and its index domain.

	Returns
	-------
	triangleEquality : bool
		Whether the relation is a SymPy `Equality` and contains a registered triangle function.
	"""
	equation, _domain = relation
	return isinstance(equation, Equality) and formulaTriangle吗(equation)

def identifyRelations(f: str, boxOfRelationsSequences: list[形Relation]) -> Iterator[tuple[str, 形Relation]]:
	"""Give each relation of the same sequences a numbered identifier.

	(AI generated docstring)

	Parameters
	----------
	f : str
		Description of the shared sequence dependencies, as produced by `make_f`.
	boxOfRelationsSequences : list[形Relation]
		Relations of those sequences, in the order used for numbering.

	Returns
	-------
	boxOfRelationsIdentified : Iterator[tuple[str, 形Relation]]
		Each relation paired with the normalized `f`, an underscore, and its ordinal from one.
		Adding a relation to a different sequence group does not change these identifiers.
	"""
	return zip(map('{}_{}'.format, repeat(makeIdentifier_f(f)), count(1)), boxOfRelationsSequences, strict=False)

def instantiateSide(side: Expr, n: int) -> Expr:
	"""Evaluate one side of a relation at an index and flatten its triangle calls.

	(AI generated docstring)

	Parameters
	----------
	side : Expr
		Symbolic expression from one side of a relation.
	n : int
		Value to substitute for `settingsOEIS.n` before evaluating sums and other operations.

	Returns
	-------
	expression : Expr
		Evaluated expression with concrete triangle cells represented by their b-file indices.
		Sequence values remain symbolic.
	"""
	return makeEquationFlat(side.subs(settingsOEIS.n, n).doit())

def instantiateRelation(identifierRelation: str, relation: 形Relation) -> Iterator[tuple[str, int, Expr, Expr]]:
	"""Instantiate a relation where every required sequence value is stored.

	(AI generated docstring)

	Parameters
	----------
	identifierRelation : str
		Identifier shared by the test parameters for this relation.
	relation : 形Relation
		Symbolic relation and its index domain. Triangle calls determine candidate indices;
		every sequence call must have a stored value after substitution and evaluation.

	Returns
	-------
	boxOfParameters : Iterator[tuple[str, int, Expr, Expr]]
		Relation identifier, index, instantiated right side, and instantiated left side, ordered
		by increasing index. A relation without two `Expr` sides of an `Equality` produces no
		parameters. Concrete triangle cells use b-file indices.
	"""
	equation, _domain = relation
	boxOfParameters: Iterator[tuple[str, int, Expr, Expr]] = iter(())
	match equation:
		case Equality(lhs=Expr() as lhs, rhs=Expr() as rhs):
			boxOf_n: tuple[int, ...] = makeIndicesRelation(relation)
			boxOfParameters = zip(repeat(identifierRelation), boxOf_n, map(partial(instantiateSide, rhs), boxOf_n), map(partial(instantiateSide, lhs), boxOf_n))
		case _:
			pass
	return boxOfParameters

def makeRelationParameters() -> tuple[tuple[str, int, Expr, Expr], ...]:
	"""Instantiate each triangle relation in `boxOfRelations` at every index whose sequence values are stored.

	(AI generated docstring)

	Each parameter holds the identifier of the relation, `n`, and the right and left sides of the instantiated relation, with each
	triangle cell represented by its b-file index. The identifier joins the sequences of the relation, as `make_f` describes them,
	with the ordinal of the relation among the relations of those sequences, so that adding a relation of other sequences changes
	no identifier.
	"""
	lookupRelations: dict[str, list[形Relation]] = toolz_groupby(compose_left(first, make_f), filter(relationTriangle吗, boxOfRelations))
	boxOfRelationsIdentified: Iterator[tuple[str, 形Relation]] = chain.from_iterable(starmap(identifyRelations, lookupRelations.items()))
	return tuple(chain.from_iterable(starmap(instantiateRelation, boxOfRelationsIdentified)))

def make_astLookupValues(oeisID: OEISid, ast_exprIndex: ast.expr) -> ast.Subscript:
	"""Represent a stored reference value as a lookup in the samples dictionary.

	(AI generated docstring)

	Parameters
	----------
	oeisID : OEISid
		Sequence whose stored value is required.
	ast_exprIndex : ast.expr
		AST expression for the sequence's b-file index.

	Returns
	-------
	astSubscript : ast.Subscript
		AST expression for `lookupValues[(oeisID, index)]`, using the supplied index expression.
	"""
	return Make.Subscript(Make.Name('lookupValues'), Make.Tuple([Make.Constant(oeisID), ast_exprIndex]))

def make_astParameterRelation(parameter: tuple[str, int, Expr, Expr]) -> ast.Call:
	"""Represent an instantiated relation test as a `pytest.param` call.

	(AI generated docstring)

	Parameters
	----------
	parameter : tuple[str, int, Expr, Expr]
		Relation identifier, index, actual expression, and expected expression. Sequence calls
		in both expressions must use concrete b-file indices.

	Returns
	-------
	astCall : ast.Call
		Call supplying the identifier, index, and both expressions with sequence values read
		from `lookupValues`. The parameter ID joins the relation identifier and index.
	"""
	identifierRelation, n, actual, expected = parameter
	return Make.Call(Make.Attribute(Make.Name('pytest'), 'param')
		, [Make.Constant(identifierRelation), Make.Constant(n)
			, make_ast_exprSymbolic(actual, make_astLookupValues), make_ast_exprSymbolic(expected, make_astLookupValues)]
		, list_keyword=[Make.keyword('id', Make.Constant(f'{identifierRelation}-{n}'))])

def make_astModuleTests(docstring: str, ingredientsFunction: IngredientsFunction) -> ast.Module:
	"""Assemble a documented test module from a function and its import dependencies.

	(AI generated docstring)

	Parameters
	----------
	docstring : str
		Documentation to place at the beginning of the generated module.
	ingredientsFunction : IngredientsFunction
		Test function and the imports needed by that function.

	Returns
	-------
	astModule : ast.Module
		Module containing the docstring, assembled imports, and test function. No file is written.
	"""
	return Make.Module([Make.Expr(Make.Constant(docstring)), *IngredientsModule(ingredientsFunction).body])

def makeTestsByFormula(boxOfParameters: tuple[tuple[OEISid, int, str | None], ...]) -> ast.Module:
	"""Build the module that tests generated OEIS formulas against stored reference values.

	(AI generated docstring)

	Parameters
	----------
	boxOfParameters : tuple[tuple[OEISid, int, str | None], ...]
		Sequence IDs, indices, and formula selectors selected by `makeFormulaParameters`.

	Returns
	-------
	astModule : ast.Module
		Module defining parametrized `test_byFormula`. Each case compares the selected
		`mapFolding.oeis.byFormula` result with its value in
		`mapFolding.tests.dataSamples.oeis.lookupValues`. No file is written or test executed.
	"""
	ingredientsFunction = IngredientsFunction(Make.FunctionDef('test_byFormula'
		, Make.arguments(list_arg=[Make.arg('oeisID', Make.Name('OEISid')), Make.arg('n', Make.Name('int'))
			, Make.arg('f', Make.BitOr.join([Make.Name('LiteralString'), Make.Constant(None)])), Make.arg('expected', Make.Name('int'))])
		, [Make.Expr(Make.Call(Make.Name('assertEqualTo')
			, [Make.Call(Make.Call(Make.Name('getattr'), [Make.Name('byFormula'), Make.Name('oeisID')]), [Make.Name('n'), Make.Name('f')])
				, Make.Name('expected'), Make.Name('oeisID'), Make.Name('n'), Make.Name('f')]))]
		, decorator_list=[Make.Call(Make.Attribute(Make.Attribute(Make.Name('pytest'), 'mark'), 'parametrize')
			, [Make.Constant('oeisID, n, f, expected'), Make.Tuple(list(map(make_astParameterFormula, boxOfParameters)))])]
		, returns=Make.Constant(None)))
	ingredientsFunction.imports.addImportFrom_asStr('__future__', 'annotations')
	ingredientsFunction.imports.addImportFrom_asStr('mapFolding.oeis', 'byFormula')
	ingredientsFunction.imports.addImportFrom_asStr('mapFolding.tests', 'assertEqualTo')
	ingredientsFunction.imports.addImportFrom_asStr('mapFolding.tests.dataSamples.oeis', 'lookupValues')
	ingredientsFunction.imports.typeChecking.addImportFrom_asStr('mapFolding.oeis', 'OEISid')
	ingredientsFunction.imports.typeChecking.addImportFrom_asStr('typing', 'LiteralString')
	ingredientsFunction.imports.addImport_asStr('pytest')
	return make_astModuleTests('Generated by mapFolding.kitAST.oeis.oeisModules.makeModulesOEIS from the symbolic relations.\n'
		'Reference: static OEIS b-file values in dataSamples.oeis.\n'
		'Counting indices, symbolic fallbacks, expanded calls, and finite sums are bounded by mapFolding.kitAST.oeis.oeisTests.'
		, ingredientsFunction)

def makeTestsTriangle(boxOfParameters: tuple[tuple[str, int, Expr, Expr], ...]) -> ast.Module:
	"""Build the module that tests triangle relations using stored sequence values.

	(AI generated docstring)

	Parameters
	----------
	boxOfParameters : tuple[tuple[str, int, Expr, Expr], ...]
		Relation identifiers, indices, right sides, and left sides from `makeRelationParameters`.

	Returns
	-------
	astModule : ast.Module
		Module defining parametrized `test_relationsTriangle`. Both sides read sequence values
		from `mapFolding.tests.dataSamples.oeis.lookupValues` and must compare equal.
		No file is written or test executed.
	"""
	ingredientsFunction = IngredientsFunction(Make.FunctionDef('test_relationsTriangle'
		, Make.arguments(list_arg=[Make.arg('relation', Make.Name('str')), Make.arg('n', Make.Name('int'))
			, Make.arg('actual', Make.BitOr.join([Make.Name('Expr'), Make.Name('int')]))
			, Make.arg('expected', Make.BitOr.join([Make.Name('Expr'), Make.Name('int')]))])
		, [Make.Expr(Make.Call(Make.Name('assertEqualTo')
			, [Make.Name('actual'), Make.Name('expected'), Make.Name('relation'), Make.Name('n')]))]
		, decorator_list=[Make.Call(Make.Attribute(Make.Attribute(Make.Name('pytest'), 'mark'), 'parametrize')
			, [Make.Constant('relation, n, actual, expected'), Make.Tuple(list(map(make_astParameterRelation, boxOfParameters)))])]
		, returns=Make.Constant(None)))
	ingredientsFunction.imports.addImportFrom_asStr('__future__', 'annotations')
	ingredientsFunction.imports.addImportFrom_asStr('mapFolding.tests', 'assertEqualTo')
	ingredientsFunction.imports.addImportFrom_asStr('mapFolding.tests.dataSamples.oeis', 'lookupValues')
	ingredientsFunction.imports.addImportFrom_asStr('sympy', 'Rational')
	ingredientsFunction.imports.addImportFrom_asStr('sympy', 'Pow')
	ingredientsFunction.imports.typeChecking.addImportFrom_asStr('sympy', 'Expr')
	ingredientsFunction.imports.addImport_asStr('pytest')
	return make_astModuleTests('Generated by mapFolding.kitAST.oeis.oeisModules.makeModulesOEIS from the symbolic relations.\n'
		'Relation: each triangle relation in `boxOfRelations` at every index whose sequence values are stored.\n'
		'Reference: static OEIS b-file values in dataSamples.oeis.\n'
		'An id joins the sequences of a relation and its ordinal among the relations of those sequences, then n.'
		, ingredientsFunction)

def getReference(call: AppliedUndef) -> tuple[OEISid, int]:
	"""Obtain the sequence ID and integer index from a symbolic sequence call.

	(AI generated docstring)

	Parameters
	----------
	call : AppliedUndef
		Call to a registered sequence function with exactly one SymPy `Integer` argument.
		Triangle cells must already be represented by their b-file indices.

	Returns
	-------
	reference : tuple[OEISid, int]
		Sequence ID and Python integer index identifying the required stored value.
	"""
	reference: tuple[OEISid, int] | None = None
	match call.args:
		case (Integer(p=index),):
			reference = (getOEISid(call), index)
		case _:
			pass
	return raiseIfNone(reference)

def getReferenceFormula(parameter: tuple[OEISid, int, str | None]) -> tuple[OEISid, int]:
	"""Obtain the reference key for a formula test's expected value.

	(AI generated docstring)

	Parameters
	----------
	parameter : tuple[OEISid, int, str | None]
		Sequence ID, index, and formula selector of a test case.

	Returns
	-------
	reference : tuple[OEISid, int]
		Sequence ID and index, independent of the formula selector.
	"""
	oeisID, n, _f = parameter
	return (oeisID, n)

def makeReferencesRelation(parameter: tuple[str, int, Expr, Expr]) -> Iterator[tuple[OEISid, int]]:
	"""Collect the stored sequence references needed by both sides of a relation test.

	(AI generated docstring)

	Parameters
	----------
	parameter : tuple[str, int, Expr, Expr]
		Relation identifier, index, actual expression, and expected expression. Each sequence
		call must use exactly one concrete integer index, including flattened triangle cells.

	Returns
	-------
	boxOfReferences : Iterator[tuple[OEISid, int]]
		Sequence IDs and indices from the distinct calls occurring on either side. Order is
		unspecified; a call shared by both sides contributes one reference.
	"""
	_identifierRelation, _n, actual, expected = parameter
	return map(getReference, actual.atoms(AppliedUndef).union(expected.atoms(AppliedUndef)))

def makeReferences(boxOfParametersFormula: tuple[tuple[OEISid, int, str | None], ...], boxOfParametersRelation: tuple[tuple[str, int, Expr, Expr], ...]
	) -> Iterator[tuple[OEISid, int]]:
	"""Collect the stored sequence references required by the generated tests.

	(AI generated docstring)

	Parameters
	----------
	boxOfParametersFormula : tuple[tuple[OEISid, int, str | None], ...]
		Sequence IDs, indices, and selectors for formula tests.
	boxOfParametersRelation : tuple[tuple[str, int, Expr, Expr], ...]
		Identifiers, indices, and both instantiated sides for relation tests.

	Returns
	-------
	boxOfReferences : Iterator[tuple[OEISid, int]]
		Formula tests' expected-value references, followed by references from both sides of
		each relation test. References repeated across test cases are retained.
	"""
	return chain(map(getReferenceFormula, boxOfParametersFormula), chain.from_iterable(map(makeReferencesRelation, boxOfParametersRelation)))

def getValueKnown(oeisID: OEISid, n: int) -> int:
	"""Obtain a stored sequence value at a specified b-file index.

	(AI generated docstring)

	Parameters
	----------
	oeisID : OEISid
		Registered sequence whose stored value is required.
	n : int
		Index present in `getValuesKnown(oeisID)`.

	Returns
	-------
	value : int
		Stored value at `n`; no missing value is computed.
	"""
	return getValuesKnown(oeisID)[n]

def make_astReference(reference: tuple[OEISid, int]) -> ast.Tuple:
	"""Represent a sequence reference as an AST tuple for a samples dictionary key.

	(AI generated docstring)

	Parameters
	----------
	reference : tuple[OEISid, int]
		Sequence ID and integer b-file index.

	Returns
	-------
	astTuple : ast.Tuple
		Tuple expression containing the sequence ID and index as literal values.
	"""
	return Make.Tuple(list(map(Make.Constant, reference)))

def makeSamplesOEIS() -> ast.Module:
	"""Build a samples module with an empty dictionary of stored sequence values.

	(AI generated docstring)

	Returns
	-------
	astModule : ast.Module
		Module containing its reference documentation, postponed annotations, and an empty
		`lookupValues` annotated as `dict[tuple[str, int], int]`. No file is written.

	See Also
	--------
	`updateSamplesOEIS`
		Add the stored reference values required by the generated tests.
	"""
	return Make.Module([Make.Expr(Make.Constant(
		'Reference: values copied from the stored OEIS b-files by makeModulesOEIS.\n'
		'Existing values are preserved; only missing sequence indices are added.'))
		, Make.ImportFrom('__future__', [Make.alias('annotations')])
		, Make.AnnAssign(Make.Name('lookupValues', Make.Store())
			, Make.Subscript(Make.Name('dict'), Make.Tuple([
				Make.Subscript(Make.Name('tuple'), Make.Tuple([Make.Name('str'), Make.Name('int')])), Make.Name('int')])), Make.Dict())])

def updateSamplesOEIS(astModule: ast.Module, boxOfReferences: Iterable[tuple[OEISid, int]]) -> ast.Module:
	"""Add missing stored sequence values to the samples module's AST.

	(AI generated docstring)

	Parameters
	----------
	astModule : ast.Module
		Samples module containing an annotated `lookupValues` assignment. Its final `ast.Dict`
		must be the literal reference dictionary, as in the module from `makeSamplesOEIS`.
	boxOfReferences : Iterable[tuple[OEISid, int]]
		Required sequence IDs and b-file indices. Every missing reference must have a stored
		value available through `getValuesKnown`.

	Returns
	-------
	astModule : ast.Module
		The supplied module with missing references appended in first-occurrence order.
		Existing keys and values retain their order and content.

	See Also
	--------
	`makeReferences`
		Collect the references required by the formula and relation tests.

	Input Mutation
	--------------
	The `lookupValues` assignment receives a new dictionary expression in the supplied module.
	Existing recorded values are preserved even when a stored value has changed. No file is written.
	"""
	astDict: ast.Dict = raiseIfNone(NodeTourist(Be.Dict, Then.extractIt).captureLastMatch(astModule))
	boxOfMissing: tuple[tuple[OEISid, int], ...] = tuple(unique(filterfalse(literal_eval(astDict).__contains__, boxOfReferences)))
	NodeChanger(Be.AnnAssign.targetIs(IfThis.isNameIdentifier('lookupValues')), Grab.valueAttribute(Then.replaceWith(Make.Dict(
		[*astDict.keys, *map(make_astReference, boxOfMissing)]
		, [*astDict.values, *map(Make.Constant, starmap(getValueKnown, boxOfMissing))])))).visit(astModule)
	return astModule
