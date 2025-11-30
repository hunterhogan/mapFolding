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
# 3, 2
# 16, 48			1/4 leavesTotal

# 7840 sequences total
# 6241    ,5,4,
# 6241    ,6,7,
# 6241    ,8,40,	1/8 leavesTotal
# 6241    ,56,24,	1/8 leavesTotal
# 5897    ,4,36,	1/16 leavesTotal
# 5897    ,9,8,
# 5889    ,10,11,
# 5889    ,52,20,	1/16 leavesTotal

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


"""
6 dimensionsTotal, with equal lengths.
a 2-dimensional plane abstracted in to 4 additional dimensions. Not a cube, hypercube, or "-hedron".
5 products of dimensions.
pile0 is a "corner".
Declare pile0 as the origin.
pile63 is a "corner", and is the "most" opposite corner from pile0.
foldsTotal is divisible by 6! Implementing this includes a side effect that leavesTotal//2 is fixed at pile63 and that indexLeaf0 is fixed at pile0.
foldsTotal is divisible by leavesTotal, so we can pin a leaf to any one pile. We pin indexLeaf0 to the origin by convention.
Implementing both of these results in indexLeaf0 pinned to pile0, indexLeaf1 fixed to pile1, and indexLeaf32 fixed to pile63.
7840 total enumerated sequences * 6! * 2^6 = 361267200 foldsTotal.

Pilings 2 and 62 are important as the first variable pilings: each pile has only 5 possible indexLeaf assignments.

The permutations of these 5 piles produce 5730 of the 7840 sequences.
2	16	32	48	62	:	5730
	16	32	48	62	:	4843
2	16	32	48		:	4843
	16	32	48		:	3425
2	16	32	  	62	:	2947
2	  	32	48	62	:	2947
2	16	  	48	62	:	2897
2	  	32	  	62	:	328
2	  		48	62	:	307
2	  			62	:	14
----------------------------
5	29	30	29	5	:	distinct indexLeaf possibilities

2	17	31	49	62	:	6068
2	17	31	47	62	:	6055
2	15	31	49	62	:	5964
2	17	31	48	62	:	5958
2	17	33	49	62	:	5863
2	15	31	47	62	:	5863
2	15	32	49	62	:	5856
2	3	4	5	6	:	134 sequences

"""

