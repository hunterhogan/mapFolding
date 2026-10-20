from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
	from collections.abc import Callable
	from typing import NotRequired

class MetadataOEISid(TypedDict):
	"""Describe known sequence values and optional triangle row settings.

	(AI generated docstring)

	You can use this dictionary schema to inspect the metadata returned by
	`mapFolding.oeis.getMetadata` [1]. The schema uses `typing.TypedDict` [2] to describe fields;
	instances remain ordinary dictionaries without runtime field validation.

	Attributes
	----------
	description : str
		Sequence description, or a diagnostic message when the description is unavailable.
	offset : int
		Starting sequence index, or a fallback sentinel when the offset is unavailable.
	valuesKnown : dict[int, int]
		Sequence indices mapped to known values.
	valueUnknown : int
		One greater than the largest key in `valuesKnown`, refreshed during metadata loading
		and local b-file updates. This value does not locate gaps below the largest stored index.
	rowLength : NotRequired[Callable[[int], int]]
		Optional callable that maps a row number to the expected complete row length.
	rowStart : NotRequired[int]
		Optional first row number. Triangle utilities use `offset` when this field is absent.

	References
	----------
	[1] `mapFolding.oeis.getMetadata`

	[2] typing.TypedDict - Python standard library.
		https://docs.python.org/3/library/typing.html#typing.TypedDict
	"""

	description: str
	"""The OEIS.org description of the integer sequence."""
	offset: int
	"""The starting index, 'n', of the sequence, typically 0 or 1."""
	valuesKnown: dict[int, int]
	"""Dictionary of sequence indices, 'n', to their known values, `totalFolds`."""
	valueUnknown: int
	"""The smallest value of 'n' for for which `totalFolds` is unknown."""

	rowLength: NotRequired[Callable[[int], int]]
	"""Obtain the expected complete row length from a row number.

	(AI generated docstring)
	"""
	rowStart: NotRequired[int]
	"""Read the optional first row number used when grouping known values.

	(AI generated docstring)
	"""
