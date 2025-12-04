# pyright: basic
# ruff: noqa
from functools import partial
from itertools import pairwise, permutations, product as CartesianProduct
from pathlib import Path
from pprint import pprint
from toolz import compose
import numpy
import pickle

fileStem = "p2d4"
pathFilename: Path = Path(f"/apps/mapFolding/Z0Z_notes/{fileStem}.csv")

headerLength = 182
headerLength = 86
headerLength = 38
data2Dn: str = pathFilename.read_text(encoding="utf-8")[headerLength:None]

def parseLineToIntegers(textLine: str) -> list[int]:
	return [int(numberText) for numberText in textLine.strip().split(",")]

listFoldings2Dn: list[list[int]] = []

for textLine in data2Dn.splitlines():
	sequenceIntegers = parseLineToIntegers(textLine)
	listFoldings2Dn.append(sequenceIntegers)

arrayFoldings2Dn = numpy.array(listFoldings2Dn, dtype=numpy.uint8)

pathFilename.with_name(f'listFoldings{fileStem}.pkl').write_bytes(pickle.dumps(listFoldings2Dn))
pathFilename.with_name(f'arrayFoldings{fileStem}.pkl').write_bytes(pickle.dumps(arrayFoldings2Dn))

dictionaryDifferencesReverse: dict[int, list[int]] = {}

for rowIndex in range(arrayFoldings2Dn.shape[0]):
	for columnIndex in range(1, arrayFoldings2Dn.shape[1]):
		valueOriginal = int(arrayFoldings2Dn[rowIndex, columnIndex])
		valueLeft = int(arrayFoldings2Dn[rowIndex, columnIndex - 1])
		differenceToLeft = valueLeft - valueOriginal

		if valueOriginal not in dictionaryDifferencesReverse:
			dictionaryDifferencesReverse[valueOriginal] = []

		if differenceToLeft not in dictionaryDifferencesReverse[valueOriginal]:
			dictionaryDifferencesReverse[valueOriginal].append(differenceToLeft)

for valueOriginal in dictionaryDifferencesReverse:
	dictionaryDifferencesReverse[valueOriginal].sort(key=abs)


pprint(dictionaryDifferencesReverse)

l5 = listLeft = [8] + [8, 16] + [2]
r4 = listRight = [8] + [8, 16]

l6 = listLeft = [8] + [8, 16]	# 2,6 is on the list of 3,2.
# However, to concatenate leafLeft,3,2,6 with 2,6,7,leafRight, leafLeft+4 == leafRight ,1,3,2,6,7,5, ,11,3,2,6,7,15, ,19,3,2,6,7,23, ,35,3,2,6,7,39,
r7 = listRight = [8] + [8, 16] + [-2]
#  7: [-2, -4, 8, 16, 32],
l=6; r=7

l8 = listLeft = [1] + [1, 2]
r40 = listRight = [1] + [1, 2] + [16]

l16 = listLeft = [1] + [1, 2, 4]
r48 = listRight = [1] + [1, 2, 4] + [-16] # 48 > 16

l56 = listLeft = [1] + [1, 2] + [-16]
r24 = listRight = [1] + [1, 2]	#24,16 is on the list of 8,40.

l9 = listLeft = [16] + [16] + [2]
r8 = listRight = [16] + [16]
l=9; r=8

l52 = listLeft = [1] + [1] + [-16]
r20 = listRight = [1] + [1]
# NOTE appends, below.
l=52; r=20
# 16, 21, 22, 28
# 36, 53, 54, 60

l4 = listLeft = [1] + [1]
r36 = listRight = [1] + [1] + [16]
l=4; r=36
# 37, 38, 32, 44, 52
# +1, +2, -4, +8, +16

l10 = listLeft = [16] + [16]
r11 = listRight = [16] + [16] + [-2]
# NOTE appends, below.
l=10; r=11
# 9,  15,  3,  27,  43
# -2, +4, -8, +16, +32

l3 = listLeft = [4] + [4, 8, 16] + [-2] # 3 > 2
r2 = listRight = [4] + [4, 8, 16]
l=3; r=2

ll = []
for index in range(1, len(listLeft)):
	ll.append(l + sum(listLeft[0:index]))
rr = []
for index in range(1, len(listRight)):
	rr.append(r + sum(listRight[0:index]))

if l > r:
	ll.append(l + listLeft[-1])
	rr.append(r + sum(listRight[0:None]))
else:
	ll.append(l + sum(listLeft[0:None]))
	rr.append(r + listRight[-1])

if (l == 4) and (r == 36):	# NOTE idk.
	rr.append(44)

if (l == 6) and (r == 7):	# 2,6 is on the list of 3,2.
	ll.append(2)

if (l == 9) and (r == 8):	# NOTE I can't explain 13. 9+4=13.
	ll.append(13)

if (l == 10) and (r == 11):	# NOTE similar to 9,8.
	rr.append(15)			# An additional, stand-alone +4. 11+4=15.
	ll.append(14)			# An additional, stand-alone +4. 10+4=14.
	ll.append(2)			# 2,10 is on the list of 3,2.

if (l == 52) and (r == 20):	# 20,16 is on the list of 16,48.
	rr.append(16)
	rr.append(28)
	ll.append(60)

if (l == 56) and (r == 24):	# 24,16 is on the list of 16,48.
	rr.append(16)

# infix: str = f",{l},{r}"
# total = 0
# for left, right in CartesianProduct(ll, rr):
# 	this: str = f",{left}{infix},{right},"
# 	count = data2Dn.count(this)
# 	total += count
# 	print(f"{count}\t{this}")

# print(total)
