# ruff: noqa
# pyright: basic
from collections import Counter
from functools import cache
from gc import collect as goByeBye, set_threshold
from tqdm import tqdm
from typing import Any
import itertools
import numpy

Z0Z_bit_lengthSafetyLimit: int = 61

type DataArray1D = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64 | numpy.signedinteger[Any]]]
type DataArray2columns = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]
type DataArray3columns = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]
type SelectorBoolean = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]]
type SelectorIndices = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.intp]]

# NOTE This code blocks enables semantic references to your data.
columnsArrayCurveGroups = columnsArrayTotal = 3
columnΩ: int = (columnsArrayTotal - columnsArrayTotal) - 1
columnDistinctCrossings = columnΩ = columnΩ + 1
columnGroupAlpha = columnΩ = columnΩ + 1
columnGroupZulu = columnΩ = columnΩ + 1
if columnΩ != columnsArrayTotal - 1:
	message = f"Please inspect the code above this `if` check. '{columnsArrayTotal = }', therefore '{columnΩ = }' must be '{columnsArrayTotal - 1 = }' due to 'zero-indexing.'"
	raise ValueError(message)
del columnsArrayTotal, columnΩ

columnsArrayCurveLocations = columnsArrayTotal = 2
columnΩ: int = (columnsArrayTotal - columnsArrayTotal) - 1
columnDistinctCrossings = columnΩ = columnΩ + 1
columnCurveLocations = columnΩ = columnΩ + 1
if columnΩ != columnsArrayTotal - 1:
	message = f"Please inspect the code above this `if` check. '{columnsArrayTotal = }', therefore '{columnΩ = }' must be '{columnsArrayTotal - 1 = }' due to 'zero-indexing.'"
	raise ValueError(message)
del columnsArrayTotal, columnΩ

groupAlphaLocator: int = 0x55555555555555555555555555555555
groupAlphaLocator64: int = 0x5555555555555555
groupZuluLocator: int = 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
groupZuluLocator64: int = 0xaaaaaaaaaaaaaaaa

def convertDictionaryCurveLocations2CurveGroups(dictionaryCurveLocations: dict[int, int]) -> dict[tuple[int, int], int]:
	return {(curveLocations & groupAlphaLocator, (curveLocations & groupZuluLocator) >> 1): distinctCrossings
		for curveLocations, distinctCrossings in dictionaryCurveLocations.items()}

@cache
def walkDyckPath(intWithExtra_0b1: int) -> int:
	"""NOTE `gc.set_threshold`: Low numbers nullify the `walkDyckPath` cache."""
	findTheExtra_0b1: int = 0
	flipExtra_0b1_Here: int = 1
	while True:
		flipExtra_0b1_Here <<= 2
		if (intWithExtra_0b1 & flipExtra_0b1_Here) == 0:
			findTheExtra_0b1 += 1
		else:
			findTheExtra_0b1 -= 1
		if findTheExtra_0b1 < 0:
			break
	return flipExtra_0b1_Here

def count(bridges: int, dictionaryCurveGroups: dict[tuple[int, int], int], bridgesMinimum: int = 0) -> tuple[int, dict[tuple[int, int], int]]:
	dictionaryCurveLocations: dict[int, int] = {}
	with tqdm(total=bridges, initial=bridges) as tqdmBar:
		# learnByExperiment = 0  # noqa: ERA001
		while (bridges > bridgesMinimum):# or any(curveGroup.bit_length() > (64 - learnByExperiment) for curveGroup in itertools.chain(*dictionaryCurveGroups.keys())):
			"""A005316, not A000682
	curveLocationsMAXIMUM: numpy.uint64 = numpy.uint64(1 << (2 * bridges + 4))
										~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
OverflowError: int too big to convert
			"""
			# if bridgesMinimum > 0 and all(curveGroup.bit_length() <= (64 - learnByExperiment) for curveGroup in itertools.chain(*dictionaryCurveGroups.keys())):  # noqa: ERA001
# NOTE NOTE NOTE NOTE HUNTER!!!!!!!! You do not understand the implications of `numpy.left_shift(arrayCurveGroups[:,
# columnGroupZulu], 3)` well enough to send 64-bit groupZulu values. Is the value effectively 67 bits now? If so, it won't fit. If
# it doesn't fit, what are the implications for the computation? You don't know.
# Furthermore, this implementation completely breaks A005316 for n >= 32.
				# break  # noqa: ERA001
			bridges -= 1
			curveLocationsMAXIMUM: int = 1 << (2 * bridges + 4)

			for (groupAlpha, groupZulu), distinctCrossings in dictionaryCurveGroups.items():
				set_threshold(0, 0, 0)  # Disable the garbage collector inside this loop to maximize the `walkDyckPath` cache hits.
				groupAlphaCurves: bool = groupAlpha > 1
				groupZuluCurves: bool = groupZulu > 1
				groupAlphaIsEven = groupZuluIsEven = 0

				# bridgesSimple
				curveLocationAnalysis = ((groupAlpha | (groupZulu << 1)) << 2) | 3
				if curveLocationAnalysis < curveLocationsMAXIMUM:
					dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

				if groupAlphaCurves:
					curveLocationAnalysis = (groupAlpha >> 2) | (groupZulu << 3) | ((groupAlphaIsEven := 1 - (groupAlpha & 1)) << 1)
					if curveLocationAnalysis < curveLocationsMAXIMUM:
						dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

				if groupZuluCurves:
					curveLocationAnalysis = (groupZulu >> 1) | (groupAlpha << 2) | (groupZuluIsEven := 1 - (groupZulu & 1))
					if curveLocationAnalysis < curveLocationsMAXIMUM:
						dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

				# bridgesAligned
				if groupAlphaCurves and groupZuluCurves and (groupAlphaIsEven or groupZuluIsEven):
					if groupAlphaIsEven and not groupZuluIsEven:
						groupAlpha ^= walkDyckPath(groupAlpha)  # noqa: PLW2901
					elif groupZuluIsEven and not groupAlphaIsEven:
						groupZulu ^= walkDyckPath(groupZulu)  # noqa: PLW2901

					curveLocationAnalysis: int = ((groupZulu >> 2) << 1) | (groupAlpha >> 2)
					if curveLocationAnalysis < curveLocationsMAXIMUM:
						dictionaryCurveLocations[curveLocationAnalysis] = dictionaryCurveLocations.get(curveLocationAnalysis, 0) + distinctCrossings

			dictionaryCurveGroups = convertDictionaryCurveLocations2CurveGroups(dictionaryCurveLocations)
			dictionaryCurveLocations = {}
			tqdmBar.update(-1)
	return (bridges, dictionaryCurveGroups)

@cache
def _flipTheExtra_0b1(avoidingLookupsInPerRowLoop: int) -> numpy.uint64:
	return numpy.uint64(avoidingLookupsInPerRowLoop ^ walkDyckPath(avoidingLookupsInPerRowLoop))

flipTheExtra_0b1 = numpy.vectorize(_flipTheExtra_0b1, otypes=[numpy.uint64])
"""NOTE NumPy docs: The vectorize function is provided primarily for convenience, not for performance. The implementation is essentially a for loop."""

def aggregateCurveLocations2CurveGroups(arrayCurveLocations: DataArray2columns) -> DataArray3columns:
	"""Deduplicate `curveLocations` by summing `distinctCrossings`; create curve groups."""
	curveLocations, indices = numpy.unique_inverse(arrayCurveLocations[:, columnCurveLocations])
	arrayCurveGroups: DataArray3columns = numpy.zeros((len(curveLocations), columnsArrayCurveGroups), dtype=numpy.uint64)
	numpy.bitwise_and(curveLocations, groupAlphaLocator64, out=arrayCurveGroups[:, columnGroupAlpha])
	numpy.bitwise_and(curveLocations, groupZuluLocator64, out=arrayCurveGroups[:, columnGroupZulu])

	curveLocations = None; del curveLocations  # noqa: E702
	goByeBye()

	arrayCurveGroups[:, columnGroupZulu] >>= 1
	numpy.add.at(arrayCurveGroups[:, columnDistinctCrossings], indices, arrayCurveLocations[:, columnDistinctCrossings])
	return arrayCurveGroups

def aggregateBridgesSimple2CurveLocations(arrayCurveLocations: DataArray2columns, indexStart: int, curveLocations: DataArray1D, distinctCrossings: DataArray1D) -> int:
	"""Deduplicate `curveLocations` by summing `distinctCrossings`."""
	miniCurveLocations, indices = numpy.unique_inverse(curveLocations[numpy.flatnonzero(curveLocations)])
	global COUNTERcurveLocations
	COUNTERcurveLocations.update(curveLocations[numpy.flatnonzero(curveLocations)])  # noqa: ERA001

	indexStop: int = indexStart + int(miniCurveLocations.size)
	arrayCurveLocations[indexStart:indexStop, columnCurveLocations] = miniCurveLocations

	miniCurveLocations = None; del miniCurveLocations  # noqa: E702
	goByeBye()

	numpy.add.at(arrayCurveLocations[indexStart:indexStop, columnDistinctCrossings], indices, distinctCrossings[numpy.flatnonzero(curveLocations)])

	return indexStop

def aggregateData2CurveLocations(arrayCurveLocations: DataArray2columns, indexStart: int, curveLocations: DataArray1D, distinctCrossings: DataArray1D, selector: SelectorBoolean, limiter: numpy.uint64) -> int:
	"""Deduplicate `curveLocations` by summing `distinctCrossings`."""
	miniCurveLocations, indices = numpy.unique_inverse(curveLocations[numpy.flatnonzero(curveLocations < limiter)])
	global COUNTERcurveLocations
	COUNTERcurveLocations.update(curveLocations[numpy.flatnonzero(curveLocations < limiter)])  # noqa: ERA001

	indexStop: int = indexStart + int(miniCurveLocations.size)
	arrayCurveLocations[indexStart:indexStop, columnCurveLocations] = miniCurveLocations

	miniCurveLocations = None; del miniCurveLocations  # noqa: E702
	goByeBye()

	numpy.add.at(arrayCurveLocations[indexStart:indexStop, columnDistinctCrossings], indices, distinctCrossings[numpy.flatnonzero(selector)[numpy.flatnonzero(curveLocations < limiter)]])

	return indexStop

def convertDictionaryCurveGroups2array(dictionaryCurveGroups: dict[tuple[int, int], int]) -> DataArray3columns:
	"""I greatly dislike this logic."""
	arrayCurveGroups: DataArray3columns = numpy.tile(numpy.fromiter(dictionaryCurveGroups.values(), dtype=numpy.uint64), (columnsArrayCurveGroups, 1)).T
	arrayCurveGroups[:, columnGroupAlpha:columnGroupZulu+1] = numpy.array(list(dictionaryCurveGroups.keys()), dtype=numpy.uint64)
	return arrayCurveGroups

def count64(bridges: int, arrayCurveGroups: DataArray3columns, bridgesMinimum: int = 0) -> tuple[int, DataArray3columns]:
	global COUNTERcurveLocations
	with tqdm(total=bridges, initial=bridges, disable=True) as tqdmBar:
		while bridges > bridgesMinimum and int(arrayCurveGroups[:, columnDistinctCrossings].max()).bit_length() < Z0Z_bit_lengthSafetyLimit:
			COUNTERcurveLocations = Counter()  # noqa: ERA001
			bridges -= 1
			curveLocationsMAXIMUM: numpy.uint64 = numpy.uint64(1 << (2 * bridges + 4))

			set_threshold(1, 1, 1)  # Re-enable the garbage collector.

			allocateGroupAlphaCurves: int = (arrayCurveGroups[:, columnGroupAlpha] > numpy.uint64(1)).sum()
			allocateGroupZuluCurves: int = (arrayCurveGroups[:, columnGroupZulu] > numpy.uint64(1)).sum()

			selectBridgesAligned: SelectorBoolean = numpy.empty_like(arrayCurveGroups[:, columnGroupAlpha], dtype=bool)
			numpy.equal(numpy.bitwise_and(arrayCurveGroups[:, columnGroupAlpha], 1), 0, out=selectBridgesAligned, dtype=bool)
			numpy.bitwise_or(selectBridgesAligned, (numpy.equal(numpy.bitwise_and(arrayCurveGroups[:, columnGroupZulu], 1), 0, dtype=bool)), out=selectBridgesAligned)
			numpy.bitwise_and(selectBridgesAligned, (arrayCurveGroups[:, columnGroupAlpha] > numpy.uint64(1)), out=selectBridgesAligned)
			numpy.bitwise_and(selectBridgesAligned, (arrayCurveGroups[:, columnGroupZulu] > numpy.uint64(1)), out=selectBridgesAligned)

			allocateBridgesAligned: int = int(numpy.count_nonzero(selectBridgesAligned))

# ----------------------------------------------- bridgesSimple -------------------------------------------------------
			curveLocationsBridgesSimpleLessThanMaximum: DataArray1D = arrayCurveGroups[:, columnGroupAlpha].copy()
			numpy.left_shift(curveLocationsBridgesSimpleLessThanMaximum, 2, out=curveLocationsBridgesSimpleLessThanMaximum)
			numpy.bitwise_or(curveLocationsBridgesSimpleLessThanMaximum, numpy.left_shift(arrayCurveGroups[:, columnGroupZulu], 3), out=curveLocationsBridgesSimpleLessThanMaximum)
			numpy.bitwise_or(curveLocationsBridgesSimpleLessThanMaximum, 3, out=curveLocationsBridgesSimpleLessThanMaximum)
			curveLocationsBridgesSimpleLessThanMaximum[curveLocationsBridgesSimpleLessThanMaximum >= curveLocationsMAXIMUM] = 0

			allocateBridgesSimple: int = int(numpy.count_nonzero(curveLocationsBridgesSimpleLessThanMaximum))

# ----------------------------------------------- arrayCurveLocations -------------------------------------------------
			rowsAllocatedTotal: int = allocateGroupAlphaCurves + allocateGroupZuluCurves + allocateBridgesSimple + allocateBridgesAligned
			arrayCurveLocations: DataArray2columns = numpy.zeros((rowsAllocatedTotal, columnsArrayCurveLocations), dtype=arrayCurveGroups.dtype)

			rowsAggregatedTotal: int = 0
			rowsDeallocatedTotal: int = 0

# ----------------------------------------------- bridgesSimple -------------------------------------------------------
			rowsAggregatedTotal = aggregateBridgesSimple2CurveLocations(arrayCurveLocations
				, rowsAggregatedTotal
				, curveLocationsBridgesSimpleLessThanMaximum
				, arrayCurveGroups[:, columnDistinctCrossings]
			)

			rowsDeallocatedTotal += allocateBridgesSimple
			arrayCurveLocations.resize((((rowsAllocatedTotal - rowsDeallocatedTotal) + rowsAggregatedTotal), columnsArrayCurveLocations))

			curveLocationsBridgesSimpleLessThanMaximum = None; del curveLocationsBridgesSimpleLessThanMaximum  # pyright: ignore[reportAssignmentType] # noqa: E702
			del allocateBridgesSimple
			goByeBye()

# ----------------------------------------------- groupAlpha ----------------------------------------------------------
			selectGroupAlphaCurves: SelectorBoolean = arrayCurveGroups[:, columnGroupAlpha] > numpy.uint64(1)
			curveLocationsGroupAlpha: DataArray1D = arrayCurveGroups[selectGroupAlphaCurves, columnGroupAlpha].copy()
			numpy.right_shift(curveLocationsGroupAlpha, 2, out=curveLocationsGroupAlpha)
			numpy.bitwise_or(curveLocationsGroupAlpha, numpy.left_shift(arrayCurveGroups[selectGroupAlphaCurves, columnGroupZulu], 3), out=curveLocationsGroupAlpha)
			numpy.bitwise_or(curveLocationsGroupAlpha, numpy.left_shift(numpy.subtract(numpy.uint64(1), numpy.bitwise_and(arrayCurveGroups[selectGroupAlphaCurves, columnGroupAlpha], 1)), 1), out=curveLocationsGroupAlpha)

			rowsAggregatedTotal = aggregateData2CurveLocations(arrayCurveLocations
				, rowsAggregatedTotal
				, curveLocationsGroupAlpha
				, arrayCurveGroups[:, columnDistinctCrossings]
				, selectGroupAlphaCurves
				, curveLocationsMAXIMUM
			)

			rowsDeallocatedTotal += allocateGroupAlphaCurves
			arrayCurveLocations.resize((((rowsAllocatedTotal - rowsDeallocatedTotal) + rowsAggregatedTotal), columnsArrayCurveLocations))

			curveLocationsGroupAlpha = None; del curveLocationsGroupAlpha  # pyright: ignore[reportAssignmentType] # noqa: E702
			del allocateGroupAlphaCurves
			selectGroupAlphaCurves = None; del selectGroupAlphaCurves # pyright: ignore[reportAssignmentType]  # noqa: E702
			goByeBye()

# ----------------------------------------------- groupZulu -----------------------------------------------------------
			selectGroupZuluCurves: SelectorBoolean = arrayCurveGroups[:, columnGroupZulu] > numpy.uint64(1)
			curveLocationsGroupZulu: DataArray1D = arrayCurveGroups[selectGroupZuluCurves, columnGroupZulu].copy()
			numpy.right_shift(curveLocationsGroupZulu, 1, out=curveLocationsGroupZulu)
			numpy.bitwise_or(curveLocationsGroupZulu, numpy.left_shift(arrayCurveGroups[selectGroupZuluCurves, columnGroupAlpha], 2), out=curveLocationsGroupZulu)
			numpy.bitwise_or(curveLocationsGroupZulu, numpy.subtract(numpy.uint64(1), numpy.bitwise_and(arrayCurveGroups[selectGroupZuluCurves, columnGroupZulu], 1)), out=curveLocationsGroupZulu)

			rowsAggregatedTotal = aggregateData2CurveLocations(arrayCurveLocations
				, rowsAggregatedTotal
				, curveLocationsGroupZulu
				, arrayCurveGroups[:, columnDistinctCrossings]
				, selectGroupZuluCurves
				, curveLocationsMAXIMUM
			)

			rowsDeallocatedTotal += allocateGroupZuluCurves
			arrayCurveLocations.resize((((rowsAllocatedTotal - rowsDeallocatedTotal) + rowsAggregatedTotal), columnsArrayCurveLocations))

			curveLocationsGroupZulu = None; del curveLocationsGroupZulu  # pyright: ignore[reportAssignmentType] # noqa: E702
			del allocateGroupZuluCurves
			selectGroupZuluCurves = None; del selectGroupZuluCurves # pyright: ignore[reportAssignmentType]  # noqa: E702
			goByeBye()

# ----------------------------------------------- bridgesAligned ------------------------------------------------------
# `bridgesAligned` = `bridgesGroupAlphaPairedToOdd` UNION WITH `bridgesGroupZuluPairedToOdd` UNION WITH `bridgesAlignedAtEven`

# bridgesAligned -------------------------------- bridgesGroupAlphaPairedToOdd ----------------------------------------
# NOTE this code block MODIFIES `arrayCurveGroups` NOTE
			set_threshold(0, 0, 0)  # Disable the garbage collector inside this loop to maximize the `walkDyckPath` cache hits.

			selectGroupAlphaAtEven: SelectorBoolean = numpy.empty_like(arrayCurveGroups[:, columnGroupAlpha], dtype=bool)
			numpy.equal(numpy.bitwise_and(arrayCurveGroups[:, columnGroupAlpha], 1), 0, out=selectGroupAlphaAtEven, dtype=bool)

			selectGroupZuluAtEven: SelectorBoolean = numpy.empty_like(arrayCurveGroups[:, columnGroupZulu], dtype=bool)
			numpy.equal(numpy.bitwise_and(arrayCurveGroups[:, columnGroupZulu], 1), 0, out=selectGroupZuluAtEven, dtype=bool)

			selectBridgesGroupAlphaPairedToOdd: SelectorIndices = numpy.flatnonzero(selectBridgesAligned & selectGroupAlphaAtEven & (~selectGroupZuluAtEven))
			arrayCurveGroups[selectBridgesGroupAlphaPairedToOdd, columnGroupAlpha] = flipTheExtra_0b1(arrayCurveGroups[selectBridgesGroupAlphaPairedToOdd, columnGroupAlpha])

			selectBridgesGroupAlphaPairedToOdd = None; del selectBridgesGroupAlphaPairedToOdd # pyright: ignore[reportAssignmentType]  # noqa: E702

# bridgesAligned -------------------------------- bridgesGroupZuluPairedToOdd ------------------------------------------
# NOTE this code block MODIFIES `arrayCurveGroups` NOTE
			set_threshold(0, 0, 0)  # Disable the garbage collector inside this loop to maximize the `walkDyckPath` cache hits.
			selectBridgesGroupZuluPairedToOdd: SelectorIndices = numpy.flatnonzero(selectBridgesAligned & (~selectGroupAlphaAtEven) & selectGroupZuluAtEven)
			arrayCurveGroups[selectBridgesGroupZuluPairedToOdd, columnGroupZulu] = flipTheExtra_0b1(arrayCurveGroups[selectBridgesGroupZuluPairedToOdd, columnGroupZulu])

			set_threshold(1, 1, 1)  # Re-enable the garbage collector.
			selectBridgesGroupZuluPairedToOdd = None; del selectBridgesGroupZuluPairedToOdd # pyright: ignore[reportAssignmentType]  # noqa: E702
			selectGroupAlphaAtEven = None; del selectGroupAlphaAtEven # pyright: ignore[reportAssignmentType]  # noqa: E702
			selectGroupZuluAtEven = None; del selectGroupZuluAtEven # pyright: ignore[reportAssignmentType]  # noqa: E702
			goByeBye()

# NOTE: All computations for `bridgesAlignedAtEven` are handled by the computations for `bridgesAligned`.

# ----------------------------------------------- bridgesAligned ------------------------------------------------------

			curveLocationsBridgesAlignedLessThanMaximum: DataArray1D = numpy.zeros((selectBridgesAligned.sum(),), dtype=numpy.uint64)
			numpy.right_shift(arrayCurveGroups[selectBridgesAligned, columnGroupZulu], 2, out=curveLocationsBridgesAlignedLessThanMaximum)
			curveLocationsBridgesAlignedLessThanMaximum <<= 1
			curveLocationsBridgesAlignedLessThanMaximum |= arrayCurveGroups[selectBridgesAligned, columnGroupAlpha] >> 2
			curveLocationsBridgesAlignedLessThanMaximum[curveLocationsBridgesAlignedLessThanMaximum >= curveLocationsMAXIMUM] = 0

			COUNTERcurveLocations.update(curveLocationsBridgesAlignedLessThanMaximum[numpy.flatnonzero(curveLocationsBridgesAlignedLessThanMaximum)])  # noqa: ERA001

			Z0Z_indexStart: int = rowsAggregatedTotal
			rowsAggregatedTotal += int(numpy.count_nonzero(curveLocationsBridgesAlignedLessThanMaximum))

			arrayCurveLocations[Z0Z_indexStart:rowsAggregatedTotal, columnCurveLocations] = curveLocationsBridgesAlignedLessThanMaximum[numpy.flatnonzero(curveLocationsBridgesAlignedLessThanMaximum)]
			arrayCurveLocations[Z0Z_indexStart:rowsAggregatedTotal, columnDistinctCrossings] = arrayCurveGroups[(numpy.flatnonzero(selectBridgesAligned)[numpy.flatnonzero(curveLocationsBridgesAlignedLessThanMaximum)]), columnDistinctCrossings]

			rowsDeallocatedTotal += allocateBridgesAligned
			arrayCurveLocations.resize((((rowsAllocatedTotal - rowsDeallocatedTotal) + rowsAggregatedTotal), columnsArrayCurveLocations))

			arrayCurveGroups = None; del arrayCurveGroups # pyright: ignore[reportAssignmentType]  # noqa: E702
			curveLocationsBridgesAlignedLessThanMaximum = None; del curveLocationsBridgesAlignedLessThanMaximum  # pyright: ignore[reportAssignmentType] # noqa: E702
			del allocateBridgesAligned
			del curveLocationsMAXIMUM
			del rowsAllocatedTotal
			del rowsDeallocatedTotal
			del Z0Z_indexStart
			del rowsAggregatedTotal
			selectBridgesAligned = None; del selectBridgesAligned  # pyright: ignore[reportAssignmentType] # noqa: E702
			goByeBye()

# ----------------------------------------------- aggregation ---------------------------------------------------------
			arrayCurveGroups = aggregateCurveLocations2CurveGroups(arrayCurveLocations)

			arrayCurveLocations = None; del arrayCurveLocations # pyright: ignore[reportAssignmentType]  # noqa: E702
			goByeBye()

			tqdmBar.update(-1)

			global oeisID, oeis_n
			print(oeisID, oeis_n, sep='\t', end='\t')  # noqa: ERA001
			# pprint(COUNTERcurveLocations)  # noqa: ERA001
			# end="\n"  # noqa: ERA001
			# print(f"{len(COUNTERcurveLocations.keys())=}", end=end) # noqa: ERA001
			# print(f"{float(numpy.average(list(COUNTERcurveLocations.values())))=}", end=end) # noqa: ERA001
			# print(f"{int(max(COUNTERcurveLocations.keys())).bit_length()=}", end=end) # noqa: ERA001
			# print(f"{int(min(COUNTERcurveLocations.keys()))=}", end=end) # noqa: ERA001
			# print(f"{COUNTERcurveLocations.total()=}", end=end) # noqa: ERA001
			# print(f"{int(COUNTERcurveLocations.most_common(1)[0][0])=}", end=end) # noqa: ERA001
			# print(f"{COUNTERcurveLocations.most_common(1)[0][1]=}", end=end) # noqa: ERA001
			# print(f"{bridges=}", end=end) # noqa: ERA001
			# print() # noqa: ERA001
			end="\t" # noqa: ERA001
			print(f"{len(COUNTERcurveLocations.keys())}", end=end) # noqa: ERA001
			print(f"{float(numpy.average(list(COUNTERcurveLocations.values())))}", end=end) # noqa: ERA001
			print(f"{int(max(COUNTERcurveLocations.keys())).bit_length()}", end=end) # noqa: ERA001
			print(f"{int(min(COUNTERcurveLocations.keys()))}", end=end) # noqa: ERA001
			print(f"{COUNTERcurveLocations.total()}", end=end) # noqa: ERA001
			print(f"{int(COUNTERcurveLocations.most_common(1)[0][0])}", end=end) # noqa: ERA001
			print(f"{COUNTERcurveLocations.most_common(1)[0][1]}", end=end) # noqa: ERA001
			print(f"{bridges}", end=end) # noqa: ERA001
			print() # noqa: ERA001

	return (bridges, arrayCurveGroups)

def convertArrayCurveGroups2dictionaryCurveGroups(arrayCurveGroups: DataArray3columns) -> dict[tuple[int, int], int]:
	return {(int(row[columnGroupAlpha]), int(row[columnGroupZulu])): int(row[columnDistinctCrossings]) for row in arrayCurveGroups}

COUNTERcurveLocations = Counter()  # noqa: ERA001

def doTheNeedful(n: int, dictionaryCurveLocations: dict[int, int]) -> int:
	"""Compute a(n) meanders with the transfer matrix algorithm.

	Parameters
	----------
	n : int
		The index in the OEIS ID sequence.
	dictionaryCurveLocations : dict[int, int]
		A dictionary mapping curve locations to their counts.

	Returns
	-------
	a(n) : int
		The computed value of a(n).

	Making sausage
	--------------

	As first computed by Iwan Jensen in 2000, A000682(41) = 6664356253639465480.
	Citation: https://github.com/hunterhogan/mapFolding/blob/main/citations/Jensen.bibtex
	See also https://oeis.org/A000682

	I'm sure you instantly observed that A000682(41) = (6664356253639465480).bit_length() = 63 bits. And A005316(44) =
	(18276178714484582264).bit_length() = 64 bits.

	If you ask NumPy 2.3, "What is your relationship with integers with more than 64 bits?"
	NumPy will say, "It's complicated."

	Therefore, to take advantage of the computational excellence of NumPy when computing A000682(n) for n > 41, I must make some
	adjustments at the total count approaches 64 bits.

	The second complication is bit-packed integers. I use a loop that starts at `bridges = n` and decrements (`bridges -= 1`)
	`until bridges = 0`. If `bridges > 29`, some of the bit-packed integers have more than 64 bits. "Hey NumPy, can I use
	bit-packed integers with more than 64 bits?" NumPy: "It's complicated." Therefore, while `bridges` is decrementing, I don't
	use NumPy until I believe the bit-packed integers will be less than 64 bits.

	A third factor that works in my favor is that peak memory usage occurs when all types of integers are well under 64-bits wide.

	In total, to compute a(n) for "large" n, I use three-stages.
	1. I use Python primitive `int` contained in a Python primitive `dict`.
	2. When the bit width of the bit-packed integers connected to `bridges` is small enough to use `numpy.uint64`, I switch to NumPy for the heavy lifting.
	3. When `distinctCrossings` subtotals might exceed 64 bits, I must switch back to Python primitives.
	"""
# NOTE `count64_bridgesMaximum = 28` is based on empirical evidence. I do not have a mathematical proof that it is correct.
	count64_bridgesMaximum = 28
	dictionaryCurveGroups: dict[tuple[int, int], int] = convertDictionaryCurveLocations2CurveGroups(dictionaryCurveLocations)

	if n >= count64_bridgesMaximum:
		n, dictionaryCurveGroups = count(n, dictionaryCurveGroups, count64_bridgesMaximum)
		goByeBye()
	# global COUNTERcurveLocations
	# COUNTERcurveLocations = Counter(dictionaryCurveLocations.keys())  # noqa: ERA001
	n, arrayCurveGroups = count64(n, convertDictionaryCurveGroups2array(dictionaryCurveGroups))
	# pprint(COUNTERcurveLocations) # noqa: ERA001
	# end="\n" # noqa: ERA001
	# print(f"{len(COUNTERcurveLocations.keys())=}", end=end) # noqa: ERA001
	# print(f"{float(numpy.average(list(COUNTERcurveLocations.values())))=}", end=end) # noqa: ERA001
	# print(f"{int(max(COUNTERcurveLocations.keys())).bit_length()=}", end=end) # noqa: ERA001
	# print(f"{int(min(COUNTERcurveLocations.keys()))=}", end=end) # noqa: ERA001
	# print(f"{COUNTERcurveLocations.total()=}", end=end) # noqa: ERA001
	# print(f"{int(COUNTERcurveLocations.most_common(1)[0][0])=}", end=end) # noqa: ERA001
	# print(f"{COUNTERcurveLocations.most_common(1)[0][1]=}", end=end) # noqa: ERA001
	# end="\t" # noqa: ERA001
	# print(f"{len(COUNTERcurveLocations.keys())}", end=end) # noqa: ERA001
	# print(f"{float(numpy.average(list(COUNTERcurveLocations.values())))}", end=end) # noqa: ERA001
	# print(f"{int(max(COUNTERcurveLocations.keys())).bit_length()}", end=end) # noqa: ERA001
	# print(f"{int(min(COUNTERcurveLocations.keys()))}", end=end) # noqa: ERA001
	# print(f"{COUNTERcurveLocations.total()}", end=end) # noqa: ERA001
	# print(f"{int(COUNTERcurveLocations.most_common(1)[0][0])}", end=end) # noqa: ERA001
	# print(f"{COUNTERcurveLocations.most_common(1)[0][1]}", end=end)  # noqa: ERA001
	# print()  # noqa: ERA001
	if n > 0:
		goByeBye()

		n, dictionaryCurveGroups = count(n, convertArrayCurveGroups2dictionaryCurveGroups(arrayCurveGroups))
		distinctCrossingsTotal: int = sum(dictionaryCurveGroups.values())
	else:
		distinctCrossingsTotal = int(arrayCurveGroups[0, columnDistinctCrossings])
	return distinctCrossingsTotal

def A000682getCurveLocations(n: int) -> dict[int, int]:
	curveLocationsMAXIMUM: int = 1 << (2 * n + 4)
	curveStart: int = 5 - (n & 0b1) * 4
	listCurveLocations: list[int] = [(curveStart << 1) | curveStart]
	while listCurveLocations[-1] < curveLocationsMAXIMUM:
		curveStart = (curveStart << 4) | 0b101
		listCurveLocations.append((curveStart << 1) | curveStart)
	return dict.fromkeys(listCurveLocations, 1)

oeisID = ""
oeis_n = 0

@cache
def A000682(n: int) -> int:
	global oeisID, oeis_n
	oeisID = "A000682"
	oeis_n = n
	return doTheNeedful(n - 1, A000682getCurveLocations(n - 1))

def A005316getCurveLocations(n: int) -> dict[int, int]:
	if n & 0b1:
		return {22: 1}
	else:
		return {15: 1}

@cache
def A005316(n: int) -> int:
	global oeisID, oeis_n
	oeisID = "A005316"
	oeis_n = n
	return doTheNeedful(n - 1, A005316getCurveLocations(n - 1))
