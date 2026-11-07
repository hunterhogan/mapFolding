from __future__ import annotations

from mapFolding.oeis._oeisdataClone import updateSnapshot
from mapFolding.oeis._theSSOT import oeisIDsImplemented, path_oeisdata, pathSnapshot

# TODO eliminate the time.txt process.
if __name__ == '__main__':
	updateSnapshot(oeisIDsImplemented, path_oeisdata, pathSnapshot)
