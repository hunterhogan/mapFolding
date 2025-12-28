"""A dump of knowledge accumulated during development."""

"""Perspective changes and code changes:

- To increment in a dimension means to add 1 in the dimensionIndex. If the current value is 0, then the new value is 1. If the
	current value is 1, then the addition requires "carrying" values to "higher" dimensionIndices.
- The `next` crease, in the sense of `k+1` and `r+1` in the inequalities is simple: given `k` and `dimension`,
	`k1 = bit_flip(k, dimension); k1 = k1 if k1 > k else None`.
	Reminder: this is because I evaluate the four `k < r`, not the eight `k ? r` inequalities *a la* Koehler.
"""

"""The 'meaning' of:
- "CTZ" or `howMany0coordinatesAtTail()` measures trailing zeros.
- `int.bit_length()` essentially measures leading zeros.
- `int.bit_count() - 1` or `howManyDimensionsHaveOddParity()` measures how many ones other than the MSD, but unlike CTZ, it counts
the absolute quantity, not just the consecutive ones relative to the LSD.
"""

"""Habits for better performance:
list:	the elements are changing, and the order matters.
tuple:	the elements are fixed, and the order matters.
set:	the elements are changing, and the order does not matter.
frozenset:	the elements are fixed, and the order does not matter.
iterator: order matters, evaluate as needed, possibly short-circuiting.
"""

"""leaf metadata:
	per dimension:
		for inequality checking:
			next leaf or None
			parity
	domain of leaf
	range of leaves in piles
"""

"""
2d6
(0, 32, 48, 56, 60, 62, 63) = sumsOfProductsOfDimensionsNearest首
(0, 1, 3, 7, 15, 31, 63, 127) = sumsOfProductsOfDimensions
leaf descends from 63 in sumsOfProductsOfDimensionsNearest首
first pile is dimensionsTotal and ascends by addends in sumsOfProductsOfDimensions
leaf63 starts at pile6 = 6+0
leaf62 starts at pile7 = 6+1
leaf60 starts at pile10 = 7+3
leaf56 starts at pile17 = 10+7
leaf48 starts at pile32 = 17+15
leaf32 starts at pile63 = 32+31

2d5
(0, 16, 24, 28, 30, 31)
31, 5+0
30, 5+1
28, 6+3
24, 9+7
16, 16+15
(0, 1, 3, 7, 15, 31, 63)
{0: [0],
 1: [1],
 2: [3, 5, 9, 17],
 3: [2, 7, 11, 13, 19, 21, 25],
 4: [3, 5, 6, 9, 10, 15, 18, 23, 27, 29],
 5: [2, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31],
 6: [3, 5, 6, 9, 10, 15, 17, 18, 23, 27, 29, 30],
 7: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31],
 8: [3, 5, 6, 9, 10, 12, 15, 18, 20, 23, 27, 29, 30],
 9: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 10: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 27, 29, 30],
 11: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 12: [3, 5, 6, 9, 10, 12, 15, 18, 20, 23, 27, 29, 30],
 13: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 14: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 27, 29, 30],
 15: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 16: [3, 5, 6, 9, 10, 12, 15, 18, 20, 23, 24, 27, 29, 30],
 17: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 18: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30],
 19: [4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 20: [5, 6, 9, 10, 12, 15, 18, 20, 23, 24, 27, 29, 30],
 21: [4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 22: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30],
 23: [4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 24: [5, 6, 9, 10, 12, 15, 18, 20, 23, 24, 27, 29, 30],
 25: [4, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 26: [9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30],
 27: [8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 28: [9, 10, 12, 18, 20, 23, 24, 27, 29, 30],
 29: [8, 19, 21, 22, 25, 26, 28],
 30: [17, 18, 20, 24],
 31: [16]}
"""


"""crazy^2
leafPredecessor = state.productsOfDimensions[dimensionNearest首(leaf)] + state.productsOfDimensions[howMany0coordinatesAtTail(leaf)]
print(leafPredecessor == int(bit_flip(0, dimensionNearest首(leaf)).bit_flip(howMany0coordinatesAtTail(leaf))))

sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = Z0Z_sumsOfProductsOfDimensionsNearest首(state)
def Z0Z_inverseIsSmallEnough(leaf: int, pile: int, sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = sumsOfProductsOfDimensionsNearest首) -> bool:
	if leaf in sumsOfProductsOfDimensionsNearest首:
		index首 = sumsOfProductsOfDimensionsNearest首.index(leaf)
		firstPile_bb = state.dimensionsTotal + sum(state.sumsOfProductsOfDimensions[0:state.dimensionsTotal-index首+inclusive])
		anotherFormula = int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin)
		print(firstPile_bb ==anotherFormula, leaf, firstPile_bb, anotherFormula)

"""

"""Pairs of leaves with low entropy.
Always consecutive:
...3, 2
...16, 48			1/4 leavesTotal

7840 sequences total
Consecutive in most sequences:
6241	...5,4,
6241	...6,7,
6241	...8,40,	1/8 leavesTotal
6241	...56,24,	1/8 leavesTotal
5897	...4,36,	1/16 leavesTotal
5897	...9,8,
5889	...10,11,
5889	...52,20,	1/16 leavesTotal

`getDomain二一零and二一` has the same basic pattern as `getDomain二零and二` with the parity switched.

Interestingly, the 22 pairs of `leaf二一, leaf二一零` in consecutive piles cover 6241 of 7840 foldsTotal for (2,) * 6 maps.
The combined domain is very small, only 76 pairs, but 22 pairs cover 80% and the other 54 pairs only cover 20%. Furthermore,
in the 22 pairs, `leaf二一零` follows `leaf二一`, but in the rest of the domain, `leaf二一` always follows `leaf二一零`.

The same for leaf二零 and leaf二, but reversed.
"""

"""General observations and analyzing pinning options.
6 dimensionsTotal, with equal lengths.
a 2-dimensional plane abstracted in to 4 additional dimensions. Not a cube, hypercube, or "-hedron".
5 products of dimensions.
pile0 is a "corner".
Declare pile0 as the origin.
pile63 is a "corner", and is the "most" opposite corner from pile0.
foldsTotal is divisible by 6! Implementing this includes a side effect that leavesTotal//2 is fixed at pile63 and that leaf0 is fixed at pile0.
foldsTotal is divisible by leavesTotal, so we can pin a leaf to any one pile. We pin leaf0 to the origin by convention.
Implementing both of these results in leaf0 pinned to pile0, leaf1 fixed to pile1, and leaf32 fixed to pile63.
7840 total enumerated sequences * 6! * 2^6 = 361267200 foldsTotal.

Pilings 2 and 62 are important as the first variable pilings: each pile has only 5 possible leaf assignments.

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
5	29	30	29	5	:	distinct leaf possibilities

2	17	31	49	62	:	6068
2	17	31	47	62	:	6055
2	15	31	49	62	:	5964
2	17	31	48	62	:	5958
2	17	33	49	62	:	5863
2	15	31	47	62	:	5863
2	15	32	49	62	:	5856
2	3	4	5	6	:	134 sequences

"""

"""Are these patterns useful?
The list of leaves in the range of pile21 for a 2d6 map:
			[2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62	],
start	step
2		12	[2, 				 14, 				 26, 				 38, 					 50,					 62	],
4		17	[	4,						 21, 							 38,												],
4		24	[	4,										 28,															 62	],
7*		6	[	   7,		 13,	 19,		 25,		 31,	 37,		NOT 43,		 49,		 55,		 61,	],
7		7	[	   7,			 14,	 21,			 28,	 35,			 42,		 49,			 56,			],
7		15	[	   7,						 22,					 37,							 52,					],
8		17	[		  8,						 25,							 42,							 59,		],
8		18	[		  8,							 26,							 44,								 62	],
11		15	[			 11,						 26,					 41,							 56,			],
11		24	[			 11,									 35,											 59,		],
14		21	[					 14,							 35,					  					 56,			],
25		12	[									 25,				 37,					 49,					 61,	],
35		12	[													 35,					 47,					 59,		],
38		6	[															 38,		 44,		 50,		 56,		 62	],
38		9	[															 38,			 47,				 56,			],
41		9	[																 41,				 50,			 59,		],
42		8	[																	 42,				 52,				 62	],
49*		6	[																				 49,		 55,		 61,	],

* If I take leaf7, step 6 to the end, it would generate leaf43, which is wrong. But, leaf49, step 6 to the end is valid.
There are many step patterns that would be valid if the ended at approximately leaf32, such as leaf19, step 3, but I only included leaf7, step 6.

[	   7, 	  11, 13, 14, 19, 21, 22, 25, 26, 28, 31, 		 38, 	 42, 44, 		 50, 52, 	 56,		 62	], # if leaf is odd, then leaf*2, leaf*4, and leaf*8 are in the range.

"""

"""products of dimensions and sums of products emerge from the formulas in `getLeafDomain`.
state = EliminationState((2,) * 6)
domainsOfDimensionOrigins = tuple(getLeafDomain(state, leaf) for leaf in state.productsOfDimensions)[0:-1]
sumsOfDimensionOrigins = tuple(accumulate(state.productsOfDimensions))[0:-1]
sumsOfDimensionOriginsReversed = tuple(accumulate(state.productsOfDimensions[::-1], initial=-state.leavesTotal))[1:None]
for dimensionOrigin, domain, sumOrigins, sumReversed in zip(state.productsOfDimensions, domainsOfDimensionOrigins, sumsOfDimensionOrigins, sumsOfDimensionOriginsReversed, strict=False):
	print(f"{dimensionOrigin:<2}\t{domain.start == sumOrigins = }\t{sumOrigins}\t{sumReversed+2}\t{domain.stop == sumReversed+2 = }")
1       domain.start == sumOrigins = True       1       2       domain.stop == sumReversed+2 = True
2       domain.start == sumOrigins = True       3       34      domain.stop == sumReversed+2 = True
4       domain.start == sumOrigins = True       7      50      domain.stop == sumReversed+2 = True
8       domain.start == sumOrigins = True       15      58      domain.stop == sumReversed+2 = True
16      domain.start == sumOrigins = True       31     62	      domain.stop == sumReversed+2 = True
32      domain.start == sumOrigins = True       63      64      domain.stop == sumReversed+2 = True

(Note to self: in `sumReversed+2`, consider if this is better explained by `sumReversed - descending + inclusive` or something similar.)

The piles of dimension origins (sums of products of dimensions) emerge from the following formulas!

(Note: the function below is included to capture the function as it existed at this point in development. I hope the package has improved/evolved by the time you read this.)
def getLeafDomain(state: EliminationState, leaf: int) -> range:
	def workhorse(leaf: int, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> range:
		originPinned =  leaf == leafOrigin
		return range(
					int(bit_flip(0, howMany0coordinatesAtTail(leaf) + 1))									# `start`, first value included in the `range`.
						+ howManyDimensionsHaveOddParity(leaf)
						- 1 - originPinned
					, int(bit_mask(dimensionsTotal) ^ bit_mask(dimensionsTotal - dimensionNearest首(leaf)))	# `stop`, first value excluded from the `range`.
						- howManyDimensionsHaveOddParity(leaf)
						+ 2 - originPinned
					, 2 + (2 * (leaf == 首零(dimensionsTotal)+零))											# `step`
				)
	return workhorse(leaf, state.dimensionsTotal, state.mapShape, state.leavesTotal)
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

I think I can use `leafInSubHyperplane` for the "Start over" equivalences.

for equate the piles:
	for equate the excluderLeaf:
		for equate the excluded leaf:
			make a rule
"""

"""Random observation about progressions within a dimension:
dimension end, 31, dimension origin, 16:
11111 	10000 16
1° crease 10010 18
31 (2, 3) 10011 19 = 18 + 1
31 (1, 3) 10101 21 = 18 + 3
31 (1, 2) 11001 25 = 18 + 7
31			< 33 = 18 + 15: not valid
sums: 0, 1, 3, 7

16 has 4 tail zeros.
16 is a dimension origin: each of the 4 zeros is a "sub-dimension origin" relative to 16.
All odd piles are covered by starting with a sub-dimension origin and adding the sums of products of dimensions.
odds: 17, 19, 21, 23, 25, 27, 29, 31
sums: 0, 1,  3,  7,  15
cf:  16, 17, 19, 23, 31
cf:  18, 19, 21, 25, NA
cf:  20, 21, 23, 27, NA
cf:  22, 23, 25, 29, NA

evens: 16, 18, 20, 22, 24, 26, 28, 30
16 is not in the table
sums: 0, 1,  3,  7,  15
cf:  17, 18, 20, 24, NA
cf:  19, 20, 22, 26, NA
cf:  21, 22, 24, 28, NA
cf:  23, 24, 26, 30, NA

To get the evens, count from the end.
evens: 16, 18, 20, 22, 24, 26, 28, 30
sums 15,  7,  3,  1, 0
cf:  16, 24, 28, 30, 31
cf:  NA, 22, 26, 28, 29
cf:  NA, 20, 24, 26, 27
cf:  NA, 18, 22, 24, 25

"""

"""Leaf precedence rules.
A hierarchy of facts: each statement is *necessarily* true about statements below it.
	Corollary: if two statements appear to contradict each other, apply the superior statement to its full scope, and apply
		the inferior statement only where it does not contradict the superior statement.
- `leafOrigin` precedes all other leaves.
- `leaf零` precedes all other leaves.
- `leaf首零` is preceded by all other leaves.

Some leaves are always preceded by one or more leaves. Most leaves, however, are preceded by one or more other leaves only if
the leaf is in a specific pile.
"""


# maps of 3 x 3 ... x 3, divisible by leavesTotal * 2^dimensionsTotal * factorial(dimensionsTotal)
