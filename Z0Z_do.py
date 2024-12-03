from mapFolding import getOEISsequence, oeisSequence_aOFn
import time
id = "A001418"
print(getOEISsequence(id))
timeStart = time.time()
print(oeisSequence_aOFn(id, 4), time.time() - timeStart)
