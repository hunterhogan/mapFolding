"""Developing elimination-based algorithms."""

from mapFolding._e._measure import (dimensionNearest首 as dimensionNearest首, dimensionSecondNearest首 as dimensionSecondNearest首, leafSubHyperplane as leafSubHyperplane,
	coordinatesOf0AtTail as coordinatesOf0AtTail, ptount as ptount)

from mapFolding._e._semiotics import 零 as 零, 一 as 一, 二 as 二, 三 as 三, 四 as 四, 五 as 五, 六 as 六, 七 as 七, 八 as 八, 九 as 九
from mapFolding._e._semiotics import 首零 as 首零, 首零二 as 首零二, 首一 as 首一, 首一二 as 首一二, 首零一 as 首零一, 首二 as 首二, 首三 as 首三, 首零一二 as 首零一二
from mapFolding._e._semiotics import decreasing as decreasing, fullRange as fullRange, leaf0 as leaf0, origin as origin

from mapFolding._e._data import (getDictionaryAddends4Next as getDictionaryAddends4Next,
	getDictionaryAddends4Prior as getDictionaryAddends4Prior,
	getDictionaryLeafDomains as getDictionaryLeafDomains,
	getDictionaryPileToLeaves as getDictionaryPileToLeaves,
    getLeafDomain as getLeafDomain)

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
