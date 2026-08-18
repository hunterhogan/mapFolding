#=SIN=
# DEVELOPMENT module.
# ruff: file-ignore[undocumented-public-class]
from __future__ import annotations

from mapFolding.algorithms.permutations import generate, initializeState, StateStampMeander
from mapFolding.theTypes import OEISid
import dataclasses

@dataclasses.dataclass(frozen=True, slots=True)
class SettingsMode:
	meanders: bool = False
	semiMeanders: bool = False
	folds: bool = False
	equivalenceClasses: bool = False
	symmetricSemiMeanders: bool = False

@dataclasses.dataclass(frozen=True, slots=True)
class SettingsGeneration:
	oeisOffset: int = 1
	Z0Z_normalizeIndex: int = 0

	mode: SettingsMode = dataclasses.field(default_factory=SettingsMode)

lookupSettings: dict[OEISid, SettingsGeneration] = {
	'A000136': SettingsGeneration(mode=SettingsMode(folds=True)),
	'A000560': SettingsGeneration(oeisOffset=2, mode=SettingsMode(symmetricSemiMeanders=True)),
	'A000682': SettingsGeneration(Z0Z_normalizeIndex=1, mode=SettingsMode(semiMeanders=True)),
	'A001011': SettingsGeneration(mode=SettingsMode(folds=True, equivalenceClasses=True)),
	'A005316': SettingsGeneration(oeisOffset=0, mode=SettingsMode(meanders=True)),
	'A077055': SettingsGeneration(Z0Z_normalizeIndex=-1, oeisOffset=0, mode=SettingsMode(meanders=True, equivalenceClasses=True)),
}

def doTheNeedful(oeisID: OEISid, n: int) -> int:
	"""Count one Sawada-Li sequence at order `n` [1].

	(AI generated docstring)

	You can use this function to select any of the six sequences implemented by Sawada and Li and
	return the number of generated objects. The function creates fresh mutable state for every call.
	The recursive core retains the paper's order convention; this boundary translates the current
	OEIS indexing, which differs by one for A000682.

	Parameters
	----------
	oeisID : OEISid
		Sequence identifier from A000136, A000560, A000682, A001011, A005316, or A077055.
	n : int
		Current OEIS index of the stamp folding, semi-meander, or open meander.

	Returns
	-------
	aOFn : int
		Number of objects or equivalence classes at OEIS index `n`.

	Raises
	------
	TypeError
		Raised when `n` is not an integer.
	ValueError
		Raised when `oeisID` is unsupported or `n` precedes the sequence's OEIS offset.

	References
	----------
	[1] Sawada, J., and Li, R. (2012). Stamp Foldings, Semi-meanders, and Open Meanders:
		Fast Generation Algorithms. The Electronic Journal of Combinatorics, 19(2), P43.
		https://doi.org/10.37236/2404
	"""
	generationMode: SettingsGeneration | None = lookupSettings.get(oeisID)
	if generationMode is None:
		message: str = f'I received `{oeisID = }`, but the Sawada-Li algorithm supports only {tuple(lookupSettings)}.'
		raise ValueError(message)
	if not isinstance(n, int) or isinstance(n, bool):
		message = f'I received `{n = }` in the form of `{type(n) = }`, but I need an integer OEIS index.'
		raise TypeError(message)
	if n < generationMode.oeisOffset:
		message = f'I received `{n = }`, but OEIS sequence `{oeisID}` is not defined below `offset = {generationMode.oeisOffset}`.'
		raise ValueError(message)

	orderSawadaLi: int = n - generationMode.Z0Z_normalizeIndex
	if orderSawadaLi == 0:
		return 1

	state: StateStampMeander = StateStampMeander(orderSawadaLi, **dataclasses.asdict(generationMode.mode))
	initializeState(state)
	generate(state, 2, 0, 0)
	return state.total
