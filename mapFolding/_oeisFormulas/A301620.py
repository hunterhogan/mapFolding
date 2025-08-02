from mapFolding._oeisFormulas.A000682 import A000682

def A301620(n: int) -> int:
	return A000682(n + 2) - 2 * A000682(n + 1)

# %F A301620 a(n) = Sum_{k=3..floor((n+3)/2)} (A259689(n+1,k)*(k-2)). - _Roger Ford_, Dec 10 2018
