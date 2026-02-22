def mapShapeIs2上nDimensions(mapShape: tuple[int, ...], *, youMustBeDimensionsTallToPinThis: int = 3) -> bool:
	"""You can test whether `mapShape` is a $2^n$-dimensional map with a configurable minimum dimension count.

	This predicate is used as a flow guard for algorithms and pinning rules that only apply to the `mapShape == (2,) * n` special
	case. The predicate returns `True` only when `len(mapShape) >= youMustBeDimensionsTallToPinThis` and each `dimensionLength` in
	`mapShape` equals `2`.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Map shape as a tuple of dimension lengths.
	youMustBeDimensionsTallToPinThis : int = 3
		Minimum number of dimensions required before treating a $2^n$-dimensional special case as eligible.

	Returns
	-------
	is2上nDimensions : bool
		`True` when `mapShape` is a $2^n$-dimensional map with the required minimum dimension count.

	Examples
	--------
	The predicate is used to gate pinning logic.

		if not mapShapeIs2上nDimensions(state.mapShape):
			return state

	The predicate is used to gate deeper special cases.

		if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=5):
			return state

	References
	----------
	[1] mapFolding._e.pin2上nDimensions.pinPilesAtEnds
		Internal package reference
	[2] mapFolding._e.dataDynamic.addLeafOptions
		Internal package reference

	"""
	return (youMustBeDimensionsTallToPinThis <= len(mapShape)) and all(dimensionLength == 2 for dimensionLength in mapShape)

