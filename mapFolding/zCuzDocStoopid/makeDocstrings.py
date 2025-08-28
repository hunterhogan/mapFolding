from mapFolding import dictionaryOEISMapFolding, dictionaryOEISMeanders
import ast

oeisID = ast.FunctionDef.name = 'A000136'
functionOf: str = 'A000682' # Or, e.g., 'A005315, A005316, and A060206'

oeisIDbyFormula= 	f"""
	Compute {oeisID}(n) as a function of {functionOf}.

	*The On-Line Encyclopedia of Integer Sequences* (OEIS) says {oeisID} is: "{dictionaryOEISMapFolding[oeisID]['description']}"

	The domain of {oeisID} starts at {dictionaryOEISMapFolding[oeisID]['offset']}, therefore for values of `n` <
	{dictionaryOEISMapFolding[oeisID]['offset']}, a(n) is undefined. The smallest value of n for which a(n) has not yet been
	computed is {dictionaryOEISMapFolding[oeisID]['valueUnknown']}.

	Parameters
	----------
	n : int
		Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and a(n) are conventions in mathematics.

	Returns
	-------
	a(n) : int
		{dictionaryOEISMapFolding[oeisID]['description']}

	Would You Like to Know More?
	----------------------------
	OEIS : webpage
		https://oeis.org/{oeisID}
	"""
