from __future__ import annotations

from typing import TypedDict

class MetadataOEISid(TypedDict):
	"""Settings for an implemented OEIS sequence."""

	description: str
	"""The OEIS.org description of the integer sequence."""
	offset: int
	"""The starting index, 'n', of the sequence, typically 0 or 1."""
	valuesKnown: dict[int, int]
	"""Dictionary of sequence indices, 'n', to their known values, `foldsTotal`."""
	valueUnknown: int
	"""The smallest value of 'n' for for which `foldsTotal` is unknown."""
