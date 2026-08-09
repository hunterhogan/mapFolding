# DEVELOPMENT module.
# pyright: reportUnusedVariable=false, reportUnusedImport=false
# ruff: file-ignore[commented-out-code, print]
"""Find `groupsOfFolds` based on Sade's 1949 insertion algorithm."""
from __future__ import annotations

from functools import partial
from hunterMakesPy import decreasing, inclusive, zeroIndexed
from itertools import chain
from mapFolding._e import leafOrigin, pileOrigin
from mapFolding._e.algorithms.iff import creaseViolation吗
from mapFolding.beDRY import defineProcessorLimit, getLeavesTotal
from mapFolding.kitFilesystem import makePathFilenameFolds, streamAlbum, writeAlbum
from mapFolding.oeis import getValuesKnown, makeMapShape
from mapFolding.theSSOT import settingsPackage
from multiprocessing.pool import Pool
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Collection, Iterable, Sequence
	from hunterMakesPy.theTypes import Limitation
	from mapFolding._e.theTypes import Folding, Leaf, LeafOptions, LeafSpace, Pile, PinnedLeaves
	from pathlib import Path

# mapShape: tuple[int, ...] = getMapNext(n)
# leavesTotal: int = getLeavesTotal(mapShape)

# def makeAlbums(getMapNext: Callable[[int], tuple[int, ...]], n: int, nFinal: int, pathAlbum: Path):
def makeAlbums1xn(n: int, nFinal: int, workersMaximum: int) -> Path:
	"""Construct every album through `nFinal`."""
	pathAlbum: Path = settingsPackage.pathPackage / '_e' / '_development' / 'albums'
	pathFilenameAlbum: Path = pathAlbum

	def excludeLeaf_rBeforeLeaf_k(folding: Folding) -> bool:
		leaf_k: Leaf = 1
		leaf_r: Leaf = 2
		return folding.index(leaf_k) < folding.index(leaf_r)

	album: Iterable[Folding] = ((leafOrigin, 1),)
	album = tuple(filter(excludeLeaf_rBeforeLeaf_k, chain.from_iterable(map(_makeDescendants, album))))

	processManager: Pool = Pool(workersMaximum)
	while n < nFinal:
		# DEVELOPMENT This line is effectively `makeMapShape('A000136', n + 1)` for the new album.
		# Figure out how to generalize to all mapShape.
		n += 1
		mapShape: tuple[int, ...] = (1, n)
		pathFilenameAlbum = makePathFilenameFolds(mapShape, pathAlbum, suffix='.album')

		if pathFilenameAlbum.exists():
			album = streamAlbum(pathFilenameAlbum)  # `streamAlbum` only read when requested: good for iterating past multiple existing files.
		else:
			album = tuple(chain.from_iterable(processManager.imap_unordered(_makeDescendants, album, chunksize=2**10)))
			writeAlbum(album, pathFilenameAlbum)

	return pathFilenameAlbum

def _makeDescendants(folding: Folding) -> tuple[Folding, ...]:
	# DEVELOPMENT With a 1Xn map, an increase in map size only adds one leaf. There are only old-n-ways to insert 1 leaf.
	# bb = len(old folding) == old leavesTotal
	# cc = len(new folding) - bb
	# If I put a Leaf in every new Pile:
	# pile permutations = P(bb, cc) = bb! / (bb - cc)!
	# If I put a LeafOptions in every new Pile:
	# each new LeafOptions = makeLeafOptions(new leavesTotal, range(bb, bb + cc))
	# pile combinations = bb Choose cc = bb! / (cc! * (bb - cc)!)
	# If I use LeafOptions, I _think_ it will be easier for me to use the functions from the
	# `elimination` algorithms. I can start with LeafOptions, which means PermutationSpace, run
	# `reduceLeafSpace` (without crossed creases, which takes most of the time), `moveToBoxOfFolding`,
	# `deconstructAtPile` on max(Pile), repeat starting at `reduceLeafSpace` until
	# boxOfPermutationSpace is empty. Then send boxOfFolding to the custom `_foldingValid吗`, which will
	# need to have dimensions added to it.

	# OR, make a new crossed creases based on the limited `_foldingValid吗` function.

	# Any mapShape can start with the 1xn Album with the same leavesTotal. But it will need to use the
	# full foldingValid check. Nevertheless, that is a MUCH smaller search space.
	inserting: Iterable[Folding] = map(partial(_insertLeafAtPile, folding, len(folding)), range(len(folding), pileOrigin, decreasing))
	return tuple(filter(_foldingValid吗, inserting))

def _insertLeafAtPile(folding: Folding, leaf: Leaf, pile: Pile) -> Folding:
	return (*folding[:pile], leaf, *folding[pile:])

def _foldingValid吗(folding: Folding) -> bool:
	# DEVELOPMENT You don't need to check every pair of creases because the existing folding is valid.
	# You just need to check new creases against same-parity-in-dimension creases.
	dictionaryLeafPile: dict[Leaf, Pile] = dict(zip(folding, range(len(folding)), strict=True))
	leafCrease: Leaf = len(folding) - 2
	pileCreasePile: tuple[Pile, Pile] = (dictionaryLeafPile[leafCrease], dictionaryLeafPile[leafCrease + 1])
	# DEVELOPMENT The step size of two is essentially hardcoding the parity of `leafComparandCrease` to match the parity of `leafCrease`.
	return not any(map(partial(_creaseViolation吗, dictionaryLeafPile, pileCreasePile), range(leafCrease - 2, leafOrigin - inclusive, 2 * decreasing)))

def _creaseViolation吗(dictionaryLeafPile: dict[Leaf, Pile], pileCreasePile: tuple[Pile, Pile], leafComparandCrease: Leaf) -> bool:
	# DEVELOPMENT With the proper sorting, you only need to check a subset of the new pairs of creases.
	# Don't sort here: sort once when the dictionary is created--maybe. It depends on how you generate the creases to be checked.
	creasesPileSorted: list[tuple[Pile, Pile]] = sorted((pileCreasePile
				, (dictionaryLeafPile[leafComparandCrease], dictionaryLeafPile[leafComparandCrease + 1])
			))
	return creaseViolation吗(creasesPileSorted[0][0], creasesPileSorted[1][0], creasesPileSorted[0][1], creasesPileSorted[1][1])

if __name__ == '__main__':
	CPUlimit: Limitation = -2
	workersMaximum: int = defineProcessorLimit(CPUlimit)
	nFinal: int = 21
	start: float = perf_counter()
	aa = makeAlbums1xn(2, nFinal, workersMaximum)
	print(f"{perf_counter() - start:.2f}")
	cc = len(aa.read_text(encoding="utf-8").splitlines()) * 2
	vv = getValuesKnown('A000682')
	print(cc == vv[nFinal + inclusive], cc, vv[nFinal + inclusive])

# DEVELOPMENT Major overhaul
# No: `Folding` -> `PinnedLeaves`. This requires overwriting keys or reading/changing a ton of keys.
# No: creaseAnte -> creasePost. ante makes it easier to count backwards from the last leaf.

# Don't create all `Folding`: create 1/2, Theorem 2, or 1/d!, Theorem 4. See mapFolding._e.algorithms.elimination theorem2b and theorem4.

# In the normal flow:
# 1. leaf r will never be inserted again, so I don't have to check if r is before k.
# 2. Important❗descendant `Folding` should also exclude leaf r before leaf k.
# Be conscious of how this is applied and watch out for traps and assumptions.

# From constraint propagation, select the length of the last dimension:
#======== Lunnon Theorem 2(b): "If some [dimensionLength in state.mapShape] > 2, [foldsTotal] is divisible by 2 * [leavesTotal]." ============================
# if (state.Theorem4Multiplier == 1) and (2 < max(state.mapShape)):
# 	state.Theorem2Multiplier = 2
# 	leafOrigin下aDimension: int = last(filter(between吗(0, state.leafLast // 2), state.productsOfDimensions))
# 	model.add(boxOfPilingsInLeafOrder[leafOrigin下aDimension] < boxOfPilingsInLeafOrder[2 * leafOrigin下aDimension])

# `leafOrigin下aDimension: int = last...` chooses the largest per-dimension leafOrigin, and I'm
# worried that value will change, such as with n X n maps.

# `elimination` selects the longest dimension:
# def theorem2b(state: EliminationState) -> EliminationState:
# 	if state.Theorem4Multiplier == 1 and (2 < max(state.mapShape)) and (4 < state.leavesTotal):
# 		state.Theorem2Multiplier = 2
# 		dimension: int = state.mapShape.index(max(state.mapShape))
# 		leaf_k: int = state.productsOfDimensions[dimension]
# 		leaf_r: int = 2 * leaf_k
# 		state = excludeLeaf_rBeforeLeaf_k(state, leaf_k, leaf_r)
# 		state = state.reduceAllPermutationSpace(boxOfFunctionsReduction).removeCreaseViolations()
# 	return state

# constraintPropagation and `elimination` would assign leaf_k = 1 because productsOfDimensions for
# mapShape (1, n) is [1, n]. (Or does my code make it [1, 1, n]?) Compare to mapShape (2, n), which is
# [1, 2, 2n]. In any event, it seems that selecting k, r, where k ∈ productsOfDimensions and r = 2 * k
# < leavesTotal is valid for any k in productsOfDimensions. With (1, n) maps, we have the good fortune
# of k and r staying the same for all n.

# If I were to select k using either of the above methods for other map types, then k and r would
# change with n. So, I would have to restore Folding that were previously excluded. For sequences with
# a static dimension length, such as (2, n) and (3, n), it seems I can choose k = the static dimension
# length, and r = 2 * k.

# I have a sense that this is not "canonical" in some way, though. I _think_ choosing the last
# dimension generates a sequence of Folding that can be sorted into "gray codes". I also think that
# multidimensional map-folding effectively generates based on the last dimension.

# On the other hand, other than Sade in 1949, no one is actually looking at the Folding leaf
# orderings.

# However, I have a strong _feeling_ that my ultra-compact notation for Folding, which is based on
# selecting the last dimension, ... wait. Theorem 2 doesn't change the folding, the implementation
# will change which half of the folding are represented. I was worried my notation wouldn't work, but
# I think it will.

# Check each crease as it is added. A violation will invalidate an entire branch. As opposed to
#   building a complete set of new `Folding` before checking any creases. At the moment, I think
#   no algorithm is _this_ efficient. "Lazy" construction allows many steps of each algorithm to be
#   efficient, but I never tried to achieve this level of efficiency.

# `leavesTotal` -> a function for incrementing the map size.
# 	E.g., getMapNext: Callable[[int], tuple[int, ...]] = partial(makeMapShape, 'A001417')

# TODO to make the files smaller, use a truncated notation. The graph notation I created is very
# compact: one delimiter and one `Leaf` represents one `Folding`, if the `Folding` are sorted. A
# `Folding` is a permutation, and there is a special notation for permutations, but it's opaque to
# me. However, there are packages that implement it, so that would likely be more robust, even if
# it is not more compact.
