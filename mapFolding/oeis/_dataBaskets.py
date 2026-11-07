from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
	from collections.abc import Callable

class MetadataOEISid(TypedDict):
	"""Describe known sequence values.

	(AI generated docstring)

	You can use this dictionary schema to inspect the metadata returned by
	`mapFolding.oeis.getMetadata`. The schema uses `typing.TypedDict` to describe fields;
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
	"""

	description: str
	"""The OEIS.org description of the integer sequence."""
	offset: int
	"""The starting index, 'n', of the sequence, typically 0 or 1."""
	valuesKnown: dict[int, int]
	"""Dictionary of sequence indices, 'n', to their known values, `totalFolds`."""
	valueUnknown: int
	"""One greater than the largest stored sequence index; gaps below this index remain possible."""

class MetadataOEISidTriangle(MetadataOEISid):
	"""Describe known sequence values and the row layout of a triangle.

	(AI generated docstring)

	Attributes
	----------
	rowLength : Callable[[int], int]
		Length of each complete row, from the row number.
	rowStart : int
		First row number. The first cell of row `rowStart` has index `offset`.
	columnStart : int
		Mathematical column index of the first cell in each row.
	"""

	rowLength: Callable[[int], int]
	"""Length of each complete row, from the row number."""
	rowStart: int
	"""First row number. The first cell of row `rowStart` has index `offset`."""
	columnStart: int
	"""Mathematical column index of the first cell in each row."""
