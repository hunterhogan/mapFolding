"""Test that beans-cornbread leaf pairs are properly maintained during pinning."""

from mapFolding._e import 一, 零, 首一, 首零一
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pin2上nDimensional import pinPile零Ante首零
import pytest

@pytest.mark.parametrize("mapShape", [
	(2, 2, 2, 2, 2, 2),
])
def test_beans_cornbread_pairing_after_pinPile零Ante首零(mapShape):
	"""Verify that pinPile零Ante首零 produces no dictionaries with beans but no cornbread.

	Beans-cornbread pairs are Gray code neighbors that must appear together:
	- (一+零, 一) = (3, 2)
	- (首一(dimensionsTotal), 首零一(dimensionsTotal)) = (16, 48) for 6 dimensions

	If leaf "beans" (3 or 16) is pinned at a pile, then leaf "cornbread" (2 or 48)
	must also be pinned somewhere in the permutationSpace (not just included in LeafOptions).
	"""
	state = EliminationState(mapShape=mapShape)
	state = pinPile零Ante首零(state, CPUlimit=False)

	leafBeansPair1 = 一 + 零
	leafCornbreadPair1 = 一
	leafBeansPair2 = 首一(state.dimensionsTotal)
	leafCornbreadPair2 = 首零一(state.dimensionsTotal)

	countBeansWithoutCornbread = 0

	for permutationSpace in state.listPermutationSpace:
		hasBeansPair1 = leafBeansPair1 in permutationSpace.values()
		hasCornbreadPair1 = leafCornbreadPair1 in permutationSpace.values()
		hasBeansPair2 = leafBeansPair2 in permutationSpace.values()
		hasCornbreadPair2 = leafCornbreadPair2 in permutationSpace.values()

		if hasBeansPair1 and not hasCornbreadPair1:
			countBeansWithoutCornbread += 1

		if hasBeansPair2 and not hasCornbreadPair2:
			countBeansWithoutCornbread += 1

	assert countBeansWithoutCornbread == 0, (
		f"pinPile零Ante首零 produced {countBeansWithoutCornbread} dictionaries with beans but no cornbread "
		f"for {mapShape=}. Expected 0. Beans-cornbread pairs: ({leafBeansPair1}, {leafCornbreadPair1}) "
		f"and ({leafBeansPair2}, {leafCornbreadPair2})."
	)
