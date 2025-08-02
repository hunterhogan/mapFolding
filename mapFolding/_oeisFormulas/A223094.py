from mapFolding._oeisFormulas.A000136 import A000136
from mapFolding._oeisFormulas.A000682 import A000682

def A223094(n: int) -> int:
    return A000136(n) - A000682(n + 1)

# %F A223094 For n >= 3: a(n) = n! - Sum_{k=3..n-1} (a(k)*n!/k!) - A000682(n+1). - _Roger Ford_, Aug 24 2024
