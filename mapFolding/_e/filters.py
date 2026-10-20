"""Test leaf assignments and filter elimination search spaces.

(AI generated docstring)

You can test leaf membership, pinned leaves, and pile positions. Use the type predicates to
distinguish a pinned leaf index from a `ChoicesLeaf` without validating index bounds.

Contents
--------
Boolean antecedents
	choicesLeaf吗
		Identify a `ChoicesLeaf` and narrow the value's type.
	leafInChoicesLeaf吗
		Test whether a `ChoicesLeaf` contains a leaf index.
	leafPinned吗
		Test whether a leaf is already assigned in a pinned-leaf mapping.
	leaf吗
		Identify a leaf-index value and narrow the value's type.
	notPileLast
		Test whether a pile differs from the final pile index.
	pileChoicesLeaf吗
		Identify a pile paired with `ChoicesLeaf` and narrow the pair's type.
Filtering functions
	是valid
		Keep permutation spaces whose validity flag is true.
"""
from __future__ import annotations

from mapFolding._e.theTypes import ChoicesLeaf
from mapFolding.theTypes import Leaf
from operator import attrgetter
from typing import TYPE_CHECKING, TypeIs

if TYPE_CHECKING:
	from mapFolding._e.dataBaskets import PermutationSpace
	from mapFolding._e.theTypes import LeafSpace, PinnedLeaves
	from mapFolding.theTypes import Pile

#======== Boolean antecedents ================================================

def leafInChoicesLeaf吗(leaf: Leaf, choicesLeaf: ChoicesLeaf) -> bool:
	"""Test whether a `ChoicesLeaf` contains a leaf index.

	(AI generated docstring)

	You can check the bit at `leaf` in `choicesLeaf` with `gmpy2.mpz.bit_test` [1]. Supply a leaf index
	below the sentinel position.

	Parameters
	----------
	leaf : Leaf
		Leaf index to test.
	choicesLeaf : ChoicesLeaf
		`ChoicesLeaf`.

	Returns
	-------
	leafIsPresent : bool
		`True` when the bit for `leaf` is set.

	References
	----------
	[1] gmpy2 integer operations and the mpz type.
		https://gmpy2.readthedocs.io/en/latest/mpz.html
	"""
	return choicesLeaf.bit_test(leaf)

def leafPinned吗(leavesPinned: PinnedLeaves, leaf: Leaf) -> bool:
	"""Test whether a leaf is already assigned in a pinned-leaf mapping.

	(AI generated docstring)

	You can check whether `leaf` occurs among the values in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PinnedLeaves
		Pinned leaf indices keyed by pile.
	leaf : Leaf
		Leaf index to find.

	Returns
	-------
	leafIsPinned : bool
		`True` when `leaf` occurs in `leavesPinned.values()`.
	"""
	return leaf in leavesPinned.values()

def notPileLast(pileLast: Pile, pile: Pile) -> bool:
	"""Test whether a pile differs from the final pile index.

	(AI generated docstring)

	You can compare `pile` with `pileLast` without checking whether `pile` belongs to the map.

	Parameters
	----------
	pileLast : Pile
		Final pile index.
	pile : Pile
		Pile index to compare.

	Returns
	-------
	pileIsNotLast : bool
		`True` when `pile` and `pileLast` differ.

	Examples
	--------
	No invocation outside tests is currently present in the codebase.
	"""
	return pileLast != pile

def leaf吗(leafSpace: LeafSpace | None) -> TypeIs[Leaf]:
	"""Identify a leaf-index value and narrow the value's type.

	(AI generated docstring)

	You can identify the integer type used for pinned leaves and use a positive result for
	`typing.TypeIs` narrowing [1] without validating index bounds. You also accept integer subclasses,
	including `bool`.

	Parameters
	----------
	leafSpace : LeafSpace | None
		Pinned leaf, `ChoicesLeaf`, or `None` to inspect.

	Returns
	-------
	leafSpaceIsLeaf : TypeIs[Leaf]
		`True` for an instance of the leaf-index type.

	See Also
	--------
	`choicesLeaf吗`
		Identify a `ChoicesLeaf` instead of a pinned leaf.
	`mapFolding._e.dataBaskets.PermutationSpace.pilePinned吗`
		Check the assignment at a pile and normalize the leaf set.

	References
	----------
	[1] Python typing.TypeIs and type narrowing.
		https://docs.python.org/3/library/typing.html#typing.TypeIs
	"""
	return isinstance(leafSpace, Leaf)

def choicesLeaf吗(leafSpace: LeafSpace | None) -> TypeIs[ChoicesLeaf]:
	"""Identify a `ChoicesLeaf` and narrow the value's type.

	(AI generated docstring)

	You can identify the `ChoicesLeaf` type and use a positive result for `typing.TypeIs` narrowing [1] without
	checking the sentinel or leaf count.

	Parameters
	----------
	leafSpace : LeafSpace | None
		Pinned leaf, `ChoicesLeaf`, or `None` to inspect.

	Returns
	-------
	leafSpaceIsChoicesLeaf : TypeIs[ChoicesLeaf]
		`True` for an instance of the `ChoicesLeaf` type.

	See Also
	--------
	`leaf吗`
		Identify a pinned leaf instead of a `ChoicesLeaf`.
	`mapFolding._e.dataBaskets.PermutationSpace.pileUndetermined吗`
		Check the assignment at a pile and normalize the leaf set.

	References
	----------
	[1] Python typing.TypeIs and type narrowing.
		https://docs.python.org/3/library/typing.html#typing.TypeIs
	"""
	return isinstance(leafSpace, ChoicesLeaf)

def pileChoicesLeaf吗(pileLeafSpace: tuple[Pile, LeafSpace]) -> TypeIs[tuple[Pile, ChoicesLeaf]]:
	"""Identify a pile paired with `ChoicesLeaf` and narrow the pair's type.

	(AI generated docstring)

	You can check that `pileLeafSpace` contains an integer pile index and a `ChoicesLeaf`, then use a positive
	result for `typing.TypeIs` narrowing [1]. Validate index bounds and the sentinel separately.

	Parameters
	----------
	pileLeafSpace : tuple[Pile, LeafSpace]
		Pile index paired with a pinned leaf or `ChoicesLeaf`.

	Returns
	-------
	pileHasChoicesLeaf : TypeIs[tuple[Pile, ChoicesLeaf]]
		`True` when the pile is an integer and the second item is a `ChoicesLeaf`.

	References
	----------
	[1] Python typing.TypeIs and type narrowing.
		https://docs.python.org/3/library/typing.html#typing.TypeIs
	"""
	return isinstance(pileLeafSpace[0], int) and choicesLeaf吗(pileLeafSpace[1])

#======== Filtering functions ================================================

def 是valid(boxOfPermutationSpace: list[PermutationSpace]) -> list[PermutationSpace]:
	"""Keep permutation spaces whose validity flag is true.

	(AI generated docstring)

	You can filter `boxOfPermutationSpace` by each object's `valid` attribute. You receive a new list
	containing the original objects without additional validation.

	Parameters
	----------
	boxOfPermutationSpace : list[PermutationSpace]
		Permutation spaces to filter.

	Returns
	-------
	boxOfPermutationSpaceValid : list[PermutationSpace]
		Original permutation-space objects with a true validity flag.
	"""
	return list(filter(attrgetter('valid'), boxOfPermutationSpace))
