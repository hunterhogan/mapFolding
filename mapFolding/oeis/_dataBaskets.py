from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
	from collections.abc import Callable
	from typing import NotRequired

class MetadataOEISid(TypedDict):
	"""Settings for an implemented OEIS sequence."""

	description: str
	"""The OEIS.org description of the integer sequence."""
	offset: int
	"""The starting index, 'n', of the sequence, typically 0 or 1."""
	valuesKnown: dict[int, int]
	"""Dictionary of sequence indices, 'n', to their known values, `totalFolds`."""
	valueUnknown: int
	"""The smallest value of 'n' for for which `totalFolds` is unknown."""

	rowLength: NotRequired[Callable[[int], int]]
	rowStart: NotRequired[int]
