import numpy

# ruff: noqa: ERA001
indexDistinctCrossings = int(0)  # noqa: RUF046, UP018
indexBifurcationAlpha = int(1)  # noqa: RUF046, UP018
indexBifurcationZulu = int(2)  # noqa: RUF046, UP018
# indexCurveLocations = imaginary(3)

def make1array(listArrays: list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]]) -> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]:
		setCurveLocations: set[numpy.uint64] = set()
		for stupidSystem in listArrays:
			setCurveLocations.update(set(stupidSystem[:, 0]))

		# `arrayOut` has `setCurveLocations`-many rows and 3 columns
		# Vector this:
		# 0 is for distinctCrossings: the sum of `distinctCrossings` with the same `curveLocations` value in any row in any array in `listArrays`
		# Vector this:
		# 1 is for bifurcationAlpha: `curveLocations` & numpy.uint64(0x5555555555555555)
		# Vector this:
		# 2 is for bifurcationZulu: `curveLocations` & numpy.uint64(0xaaaaaaaaaaaaaaaa)) >> numpy.uint64(1)
		# 3 does not exist: if we had infinite memory, then 3 would be for `curveLocations`. We don't have enough memory, so we precompute the bifurcations and discard `curveLocations`.

		return arrayOut # The next `arrayCurveLocations`

def convertDictionaryToNumPy(dictionaryIn: dict[int, int]) -> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]:
	arrayKeys: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.fromiter(dictionaryIn.keys(), dtype=numpy.uint64)

	return numpy.column_stack((
		numpy.fromiter(dictionaryIn.values(), dtype=numpy.uint64)
		, arrayKeys & numpy.uint64(0x5555555555555555)
		, (arrayKeys & numpy.uint64(0xaaaaaaaaaaaaaaaa)) >> numpy.uint64(1)
	))

def count(bridges: int, startingCurveLocations: dict[int, int]) -> int:
	arrayCurveLocations: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = convertDictionaryToNumPy(startingCurveLocations)

	while bridges > 0:
		bridges -= 1

		IDKlistArrayCurveLocationsAnalyzedIDK: list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]]] = []

		# Vector conditional
		curveLocationsMAXIMUM: numpy.uint64 = numpy.uint64(1) << numpy.uint64(2 * bridges + 4)

		# selectSimpleBridges
		curveLocationsSimpleBridges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (
			((arrayCurveLocations[:, indexBifurcationAlpha] | (arrayCurveLocations[:, indexBifurcationZulu] << numpy.uint64(1))) << numpy.uint64(2)) | numpy.uint64(3))
		IDKlistArrayCurveLocationsAnalyzedIDK.append(numpy.column_stack((
			arrayCurveLocations[curveLocationsSimpleBridges < curveLocationsMAXIMUM, indexDistinctCrossings]
			, curveLocationsSimpleBridges[curveLocationsSimpleBridges < curveLocationsMAXIMUM]
		)))

		selectBifurcationAlphaCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveLocations[:, indexBifurcationAlpha] > numpy.uint64(1)

		curveLocationsBifurcationAlphaCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.bitwise_or.reduce((
			arrayCurveLocations[selectBifurcationAlphaCurves, indexBifurcationAlpha] >> numpy.uint64(2),
			arrayCurveLocations[selectBifurcationAlphaCurves, indexBifurcationZulu] << numpy.uint64(3),
			(numpy.uint64(1) - (arrayCurveLocations[selectBifurcationAlphaCurves, indexBifurcationAlpha] & numpy.uint64(1))) << numpy.uint64(1)
		), axis=0)

		IDKlistArrayCurveLocationsAnalyzedIDK.append(numpy.column_stack((
			arrayCurveLocations[selectBifurcationAlphaCurves, indexDistinctCrossings][curveLocationsBifurcationAlphaCurves < curveLocationsMAXIMUM]
			, curveLocationsBifurcationAlphaCurves[curveLocationsBifurcationAlphaCurves < curveLocationsMAXIMUM]
		)))

		selectBifurcationZuluCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveLocations[:, indexBifurcationZulu] > numpy.uint64(1)

		curveLocationsBifurcationZuluCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = numpy.bitwise_or.reduce((
			arrayCurveLocations[selectBifurcationZuluCurves, indexBifurcationZulu] >> numpy.uint64(1)
			, arrayCurveLocations[selectBifurcationZuluCurves, indexBifurcationAlpha] << numpy.uint64(2)
			, ~ (arrayCurveLocations[selectBifurcationZuluCurves, indexBifurcationZulu] & numpy.uint64(1))
		), axis=0)

		IDKlistArrayCurveLocationsAnalyzedIDK.append(numpy.column_stack((
			arrayCurveLocations[selectBifurcationZuluCurves, indexDistinctCrossings][curveLocationsBifurcationZuluCurves < curveLocationsMAXIMUM]
			, curveLocationsBifurcationZuluCurves[curveLocationsBifurcationZuluCurves < curveLocationsMAXIMUM]
		)))

		# Defining selectors
		selectBifurcationAlphaEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveLocations[:, indexBifurcationAlpha] & numpy.uint64(1)) == numpy.uint64(0)
		selectBifurcationZuluEven: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveLocations[:, indexBifurcationZulu] & numpy.uint64(1)) == numpy.uint64(0)
		selectBifurcationsEven = numpy.logical_or.reduce((selectBifurcationAlphaEven, selectBifurcationZuluEven))
		Z0Z_selectBridgesAlignedAll: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.logical_and.reduce((selectBifurcationAlphaCurves, selectBifurcationZuluCurves, selectBifurcationsEven))
		Z0Z_selectBridgesBifurcationAlphaToRepair: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.logical_and.reduce((Z0Z_selectBridgesAlignedAll, ~selectBifurcationZuluEven))
		Z0Z_selectBridgesBifurcationZuluToRepair: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.logical_and.reduce((Z0Z_selectBridgesAlignedAll, ~selectBifurcationAlphaEven))

		# Z0Z_bifurcationAlphaPaired
			# Initialize
		Z0Z_bifurcationAlphaPairedRemainingToModify = Z0Z_selectBridgesBifurcationAlphaToRepair.copy()
		XOrHere2makePair = numpy.uint64(1)
		selectToModifyByXOR: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.array([])
		selectorUnified: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.array([])

		while Z0Z_bifurcationAlphaPairedRemainingToModify.any():
			XOrHere2makePair <<= numpy.uint64(2)

			# New condition
			selectToModifyByXOR: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveLocations[:, indexBifurcationAlpha] & XOrHere2makePair) == numpy.uint64(0)
			selectorUnified: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.logical_and.reduce((Z0Z_bifurcationAlphaPairedRemainingToModify, selectToModifyByXOR))

			# Modify in place
			arrayCurveLocations[selectorUnified, indexBifurcationAlpha] = (arrayCurveLocations[selectorUnified, indexBifurcationAlpha] ^ XOrHere2makePair) >> numpy.uint64(2)

			# Remove the modified elements
			Z0Z_bifurcationAlphaPairedRemainingToModify: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.logical_and.reduce((Z0Z_bifurcationAlphaPairedRemainingToModify, ~selectorUnified))

		# Z0Z_bifurcationZuluPaired
			# Initialize
		Z0Z_selectBifurcationZuluToModify: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = Z0Z_selectBridgesBifurcationZuluToRepair.copy()
		XOrHere2makePair: numpy.uint64 = numpy.uint64(1)
		selectToModifyByXOR: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.array([])
		selectorUnified: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.array([])

		while Z0Z_selectBifurcationZuluToModify.any():
			XOrHere2makePair <<= numpy.uint64(2)

			# New condition
			selectUnpaired_0b1: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = (arrayCurveLocations[:, indexBifurcationZulu] & XOrHere2makePair) == numpy.uint64(0)
			selectorUnified: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.logical_and.reduce((Z0Z_selectBifurcationZuluToModify, selectUnpaired_0b1))

			# Modify in place
			arrayCurveLocations[selectorUnified, indexBifurcationZulu] ^= XOrHere2makePair

			# Remove the modified elements from the selector
			Z0Z_selectBifurcationZuluToModify: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = numpy.logical_and.reduce((Z0Z_selectBifurcationZuluToModify, ~selectorUnified))

		curveLocation_Z0Z_allAlignedBridges: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint64]] = (
			((arrayCurveLocations[Z0Z_selectBridgesAlignedAll, indexBifurcationZulu] >> numpy.uint64(2)) << numpy.uint64(1))
			| (arrayCurveLocations[Z0Z_selectBridgesAlignedAll, indexBifurcationAlpha] >> numpy.uint64(2))
		)

		IDKlistArrayCurveLocationsAnalyzedIDK.append(numpy.column_stack((
			arrayCurveLocations[Z0Z_selectBridgesAlignedAll, indexDistinctCrossings][curveLocation_Z0Z_allAlignedBridges < curveLocationsMAXIMUM]
			, curveLocation_Z0Z_allAlignedBridges[curveLocation_Z0Z_allAlignedBridges < curveLocationsMAXIMUM]
		)))

		arrayCurveLocations = make1array(IDKlistArrayCurveLocationsAnalyzedIDK)

		print(sum(arrayCurveLocations[:, 0]))  # noqa: T201

		IDKlistArrayCurveLocationsAnalyzedIDK.clear()

	return int(sum(arrayCurveLocations[:, 0]))

