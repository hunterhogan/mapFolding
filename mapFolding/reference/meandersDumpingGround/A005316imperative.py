def count(bridges: int, dictionaryStateToTotal: dict[int, int]) -> int:  # noqa: C901, PLR0912
    while bridges > 0:
        bridges -= 1
        archStateLimit = 1 << (2 + (2 * (bridges + 1)))

        dictionaryStateToTotalNext: dict[int, int] = {}

        for state, totalCurrent in dictionaryStateToTotal.items():
            maskBits = 0x5555555555555555
            if state > maskBits:
                bitWidth = 64
                while maskBits < state:
                    maskBits |= maskBits << bitWidth
                    bitWidth <<= 1

            lower = state & maskBits
            upper = (state ^ lower) >> 1

            listNextStates: list[int] = []

            lower_not_one = lower != 1
            upper_not_one = upper != 1
            lower_even = (lower & 1) == 0
            upper_even = (upper & 1) == 0

            if lower_not_one:
                nextState = (lower >> 2) | (((upper << 2) ^ (1 if lower_even else 0)) << 1)
                if nextState < archStateLimit:
                    listNextStates.append(nextState)

            if upper_not_one:
                nextState = ((lower << 2) ^ (1 if upper_even else 0)) | ((upper >> 2) << 1)
                if nextState < archStateLimit:
                    listNextStates.append(nextState)

            nextState = ((lower << 2) | 1) | (((upper << 2) | 1) << 1)
            if nextState < archStateLimit:
                listNextStates.append(nextState)

            if lower_not_one and upper_not_one and (lower_even or upper_even):
                temp_lower, temp_upper = lower, upper

                if lower_even and not upper_even:
                    archBalance = 0
                    bitPosition = 1
                    while archBalance >= 0:
                        bitPosition <<= 2
                        archBalance += 1 if (temp_lower & bitPosition) == 0 else -1
                    temp_lower ^= bitPosition

                if upper_even and not lower_even:
                    archBalance = 0
                    bitPosition = 1
                    while archBalance >= 0:
                        bitPosition <<= 2
                        archBalance += 1 if (temp_upper & bitPosition) == 0 else -1
                    temp_upper ^= bitPosition

                nextState = (temp_lower >> 2) | ((temp_upper >> 2) << 1)
                if nextState < archStateLimit:
                    listNextStates.append(nextState)

            for nextState in listNextStates:
                dictionaryStateToTotalNext[nextState] = dictionaryStateToTotalNext.get(nextState, 0) + totalCurrent

        dictionaryStateToTotal = dictionaryStateToTotalNext

    return sum(dictionaryStateToTotal.values())

def initializeA005316(remainingBridges: int) -> dict[int, int]:
	bridgesTotalIsOdd = (remainingBridges & 1) == 1
	if bridgesTotalIsOdd:
		arrayBitPattern = (1 << 2) | 1
		arrayBitPattern <<= 2
		initialState = arrayBitPattern | 1 << 1
		return {initialState: 1}
	else:
		arrayBitPattern = (1 << 2) | 1
		initialState = arrayBitPattern | arrayBitPattern << 1
		return {initialState: 1}

def initializeA000682(remainingBridges: int) -> dict[int, int]:
	bridgesTotalIsOdd = (remainingBridges & 1) == 1
	archStateLimit = 1 << (2 + (2 * (remainingBridges + 1)))

	dictionaryStateToTotal: dict[int, int] = {}
	arrayBitPattern = 1 if bridgesTotalIsOdd else ((1 << 2) | 1)

	arrayPackedState = arrayBitPattern | arrayBitPattern << 1
	while arrayPackedState < archStateLimit:
		dictionaryStateToTotal[arrayPackedState] = 1
		arrayBitPattern = ((arrayBitPattern << 2) | 1) << 2 | 1
		arrayPackedState = arrayBitPattern | arrayBitPattern << 1

	return dictionaryStateToTotal

def A005316(n: int) -> int:
	return count(n, initializeA005316(n))

def A000682(n: int) -> int:
	return count(n - 1, initializeA000682(n - 1))
