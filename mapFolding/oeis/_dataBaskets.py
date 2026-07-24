from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
	from collections.abc import Callable

class MetadataOEISidMapFoldingManuallySet(TypedDict):
	"""Settings that are best selected by a human instead of algorithmically."""

	getMapShape: Callable[[int], tuple[int, ...]]
	"""Function to convert the OEIS sequence index, 'n', to its `mapShape` tuple."""

class MetadataOEISidMapFolding(TypedDict):
	"""Settings for an OEIS ID that may be computed by a multidimensional map folding algorithm."""

	description: str
	"""The OEIS.org description of the integer sequence."""
	getMapShape: Callable[[int], tuple[int, ...]]
	"""Function to convert the OEIS sequence index, 'n', to its `mapShape` tuple."""
	offset: int
	"""The starting index, 'n', of the sequence, typically 0 or 1."""
	valuesKnown: dict[int, int]
	"""Dictionary of sequence indices, 'n', to their known values, `foldsTotal`."""
	valueUnknown: int
	"""The smallest value of 'n' for for which `foldsTotal` is unknown."""

class MetadataOEISidManuallySet(TypedDict, total=False):
	"""Placeholder for future manually curated OEIS metadata."""

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
