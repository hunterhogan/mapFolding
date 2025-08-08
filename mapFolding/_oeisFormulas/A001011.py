from mapFolding._oeisFormulas.A000136 import A000136
from mapFolding._oeisFormulas.A001010 import A001010

def A001011(n: int) -> int:
    return (A001010(n) + A000136(n)) // 4
