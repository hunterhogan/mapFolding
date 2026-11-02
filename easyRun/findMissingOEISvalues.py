from __future__ import annotations

from ast import fix_missing_locations
from astToolkit import Make
from mapFolding.kitAST import IngredientsModule
from mapFolding.kitAST.oeisIDByFormula import makeSymbolicFormula
from mapFolding.oeis import getMetadata, updateBFile
from mapFolding.oeis._symbolic import getMissingFormulas
from operator import itemgetter
from types import ModuleType

if __name__ == '__main__':
	keepGoing: bool = True
	while keepGoing:
		keepGoing = False
		#=SIN= For loop: attempt every callable on each pass; ascending `n` lets one pass extend a b-file by several terms.
		for oeisID, n, expression in sorted(getMissingFormulas(), key=itemgetter(1)):
			if n != getMetadata(oeisID)['valueUnknown']:
				continue
			ingredientsFunction = makeSymbolicFormula(oeisID, expression)
			module = ModuleType(oeisID)
			exec(compile(fix_missing_locations(Make.Module(IngredientsModule(ingredientsFunction).body)), __file__, 'exec'), module.__dict__)  # ruff: ignore[exec-builtin]
			countTotal: int = getattr(module, oeisID)()
			updateBFile(oeisID, {n: countTotal})
			keepGoing = True
			print(oeisID, n, expression, countTotal)
