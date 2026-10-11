from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
	from collections.abc import Callable
	from typing import NotRequired

class MetadataOEISid(TypedDict):
	"""Describe known sequence values and optional triangle row settings.

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
		One greater than the largest key in `valuesKnown`, as assigned during metadata loading.
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
	"""Read the sequence description or a diagnostic message when no description is available."""
	offset: int
	"""Read the starting sequence index or a fallback sentinel when no offset is available."""
	valuesKnown: dict[int, int]
	"""Look up known sequence values by integer index."""
	valueUnknown: int
	"""Read one greater than the largest known index, without checking for gaps in the indices."""

	rowLength: NotRequired[Callable[[int], int]]
	"""Obtain the expected complete row length from a row number.

	(AI generated docstring)
	"""
	rowStart: NotRequired[int]
	"""Read the optional first row number used when grouping known values.

	(AI generated docstring)
	"""
