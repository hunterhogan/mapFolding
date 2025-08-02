from mapFolding._oeisFormulas.A000682 import A000682

def A000560(n: int) -> int:
    return A000682(n + 1) // 2
