# ruff: ignore[undocumented-public-module]
# DOCUMENT
from __future__ import annotations

from research.matrixMeanders.formulasTriangle import boxOfDiagonals
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable
	from mapFolding.dataBaskets import StateMeanders

def shortcut(state: StateMeanders) -> StateMeanders:  # ruff: ignore[undocumented-public-function]
	# DOCUMENT
	boundary: int = state.boundary + 1

	def removeKnownValue(次diagonal: int, 工: Callable[[int], int]) -> int:
		arcCode: int = (1 << (2 * (boundary - 2 * 次diagonal) + 2)) - 1
		subtotalMeanders: int = state.lookupMeanders.pop(arcCode, 0)
		return 工(boundary) * subtotalMeanders

	state.n += sum(map(removeKnownValue, range(1, boundary // 2 + 1), boxOfDiagonals))
	if not state.lookupMeanders:
		state.boundary = 0
	return state
