"""Developing elimination-based algorithms."""

from mapFolding._e._measure import (dimensionNearest首 as dimensionNearest首, dimensionSecondNearest首 as dimensionSecondNearest首, leafInSubHyperplane as leafInSubHyperplane,
	howMany0coordinatesAtTail as howMany0coordinatesAtTail, ptount as ptount)

from mapFolding._e._semiotics import 零 as 零, 一 as 一, 二 as 二, 三 as 三, 四 as 四, 五 as 五, 六 as 六, 七 as 七, 八 as 八, 九 as 九
from mapFolding._e._semiotics import 首零 as 首零, 首零二 as 首零二, 首一 as 首一, 首一二 as 首一二, 首零一 as 首零一, 首二 as 首二, 首三 as 首三, 首零一二 as 首零一二
from mapFolding._e._semiotics import leafOrigin as leafOrigin, pileOrigin as pileOrigin, PinnedLeaves as PinnedLeaves

from mapFolding._e._data import (getDictionaryAddends4Next as getDictionaryAddends4Next,
	getDictionaryAddends4Prior as getDictionaryAddends4Prior,
	getDictionaryLeafDomains as getDictionaryLeafDomains,
	getDictionaryPileToLeaves as getDictionaryPileToLeaves,
    getLeafDomain as getLeafDomain,
    getListLeavesIncrease as getListLeavesIncrease,
	getListLeavesDecrease as getListLeavesDecrease
	)


"""
leaf1 is a dimension origin: its addends up to [-1], which equate to leaves 3, 5, 9, 17, come before the dimension origins, 2, 4, 8, 16.

This is due to:
leaf	{dimension: increase}
0			{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32} <- dimension origins
1			{1: 3, 2: 5, 3: 9, 4: 17, 5: 33}

If leaf2 were before leaf3, it would interpose the crease from leaf1 to leaf3 in dimension1.

Similarly, leaf2 addends up to [-1], which equate to leaves 6, 10, 18 come before dimension origins, 4, 8, 16.
2			{0: 3, 2: 6, 3: 10, 4: 18, 5: 34}

The rule against interposing is so strong it extends to leaf3, which is not a dimension origin, but is the first increase from leaf1.
leaf3 addends up to [-1], which equate to leaves 7, 11, 19, come before the dimension origins, 4, 8, 16.
3			{2: 7, 3: 11, 4: 19, 5: 35}

leaf4 is the dimension2 origin and its increases 12 and 20 come before dimension origins 8 and 16.
4			{0: 5, 1: 6, 3: 12, 4: 20, 5: 36}

leaf5, 0b101, 二 + 零, which absolutely has the coordinates of 1 in dimension2, 二, and 1 in dimension0, 零, comes before all multiples of 4.

leaf6, 二 + 一, is the same as leaf5.

leaf7, 二 + 一 + 零, is also the same as leaf5 and leaf6!

leaf9, 三 + 零, comes before the dimension3 origin leaf8, as described above, and before all multiples of 8, or 三.

Furthermore, all leaves between 三+零 and 三+二+一+零, inclusive, come before 三 (8) and its multiples.

The same thing happens at the next dimension, 四. leaves 17-31 all come before 16, 32, and 48. This example is a 6 dimensional
map. Because all leaves less than 32 must come before leaf32, it cannot appear before pile 32. It's fixed at the last pile, of course.

wow.

if `dimensionNearest首(k) <= coordinatesOf0AtTail(r)`, then must `pileOf_k < pileOf_r`
"""


"""Equating pile = leavesTotal // 2 - 1.
dict_keys([2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31])
dict_keys([2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31, 35, 37, 38, 41, 42, 47, 49, 50, 55, 59, 61, 62])

dict_keys([2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31												  ]) # bit_mask(5)
dict_keys([	  2,    		 7, 		11, 	13,     		19, 	21,     	25,     		31]) # right shift 1 or //2
dict_keys([	     											26,	25,	22,	21,		14,	13,		 4,	 2,	  ]) # count from the end: 63 - x.
		1,			11, 13,		19, 21,		25,			35, 37,		41,			49,		55, 59
		31,			26, 25,		22, 21,		19,			14, 13,		11,			 7,		 4,  2
dict_keys([			26, 25,		22, 21, 	19,			14, 13,		11,			 7,		 4,	 2,		  ]) # Double and count from the end.
dict_keys([						11, 13, 14															  ]) # "Start over" at leavesTotal // 4.
dict_keys([												19, 21, 22, 25, 26, 31						  ]) # "Start over" at leavesTotal // 2.
dict_keys([																		25, 26, 31			  ]) # "Start over" at 3 * leavesTotal // 4.

"Counting from the end" is a permutation of the indices applied by bit manipulation.
	pile31 if leavesTotal=64 (see above):
dict_keys([2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31, 35, 37, 38, 41, 42, 47, 49, 50, 55, 59, 61, 62])
	pile15 if leavesTotal=32, but mapping values based on "counting from the end" (see above):
dict_keys([	     											26,	25,	22,	21,		14,	13,		 4,	 2,	  ]) # count from the end: 63 - x.
	permute by flipping the bits (see below):
														... 26, 25, 22, 21, ... 14, 13,	...	 4,  2
														... 37, 38, 41, 42, ... 49, 50,	...	59, 61

from gmpy2 import bit_mask
for nn in [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31][::-1]:
	ii = 6
	mm = bit_mask(ii)
	invert = nn ^ mm
	print(nn, invert, sep='\t-> ')

bin(63) 	= 0b111111
bit_mask(6) = 0b111111

31	-> 32
26	-> 37
25	-> 38
22	-> 41
21	-> 42
19	-> 44
14	-> 49
13	-> 50
11	-> 52
7	-> 56
4	-> 59
2	-> 61

o.m.f.g.

for equate the piles:
	for equate the excluderLeaf:
		for equate the excluded leaf:
			make a rule
"""

"""NOTE
multiplicityOfPrimeFactor2 measures trailing zeros.
.bit_length() essentially measures leading zeros.

.bit_count() -1 measures ones, and the ones must be in the digits that trail the MSD.

"""
