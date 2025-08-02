from mapFolding._oeisFormulas.A000682 import A000682

def A259702(n: int) -> int:
	return A000682(n) // 2 - A000682(n - 1)
