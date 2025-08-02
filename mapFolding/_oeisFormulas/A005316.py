from mapFolding._oeisFormulas.matrixMeanders import count

def initializeA005316(n: int) -> dict[int, int]:
	if n & 0b1:
		return {22: 1}
	else:
		return {15: 1}

def A005316(n: int) -> int:
	return count(n-1, initializeA005316(n-1))
