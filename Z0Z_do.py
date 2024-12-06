from mapFolding import oeisSequence_aOFn
from mapFolding.oeis import _getOEISsequence
import time
id = "A195646"
print(_getOEISsequence(id))
timeStart = time.perf_counter()
print(oeisSequence_aOFn(id, 3), time.perf_counter() - timeStart)
