from mapFolding import getOEISsequence, oeisSequence_aOFn
import time
id = "A195646"
print(getOEISsequence(id))
timeStart = time.perf_counter()
print(oeisSequence_aOFn(id, 3), time.perf_counter() - timeStart)
