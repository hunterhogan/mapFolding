from __future__ import annotations

from typing import cast as ILiterallyPromiseLiteralLiterallyMeansLiteral, LiteralString, TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid

def formatOEISid(oeisID: OEISid) -> OEISid:
	"""I use this to normalize OEIS sequence identifiers to a canonical form.

	This shared normalization function ensures consistent OEIS sequence ID formatting across all
	retrieval, lookup, and computation operations throughout the module. The function converts the
	identifier to uppercase and removes leading and trailing whitespace to ensure reliable dictionary
	lookups and cache key formation.

	Parameters
	----------
	oeisID : OEISid
		The OEIS sequence identifier to standardize.

	Returns
	-------
	oeisIDstandardized : OEISid
		Uppercase, alphanumeric OEIS ID with no whitespace.

	"""
	return ILiterallyPromiseLiteralLiterallyMeansLiteral("LiteralString", str(oeisID).upper().strip())
