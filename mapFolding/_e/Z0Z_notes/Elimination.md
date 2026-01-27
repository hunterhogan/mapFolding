# What I think I know about my "elimination" algorithm

**Important**: The last section of this document is a "glossary" with critical contextual details. You ought to skim it now and consult it as needed.

## Rules for maintaining a valid permutation space

1. In `permutationSpace`, if `leaf` is not pinned, deconstruct `permutationSpace` by the `Pile` domain of `leaf`.
   1. For each `pile` in the domain of `leaf`, if `pile` in `permutationSpace` is not occupied, create a new `PermutationSpace` dictionary by appending `leaf` pinned at `pile` to `permutationSpace`.
   2. Replace `permutationSpace` with the group of newly created `PermutationSpace` dictionaries.
2. In `permutationSpace`, if a `pile` is not pinned, deconstruct `permutationSpace` by the `Leaf` range (mathematical range) of `pile`.
   1. For each `leaf` in the range of `pile`, if `leaf` is not already pinned in `permutationSpace`, create a new `PermutationSpace` dictionary by appending `leaf` pinned at `pile` to `permutationSpace`.
   2. Replace `permutationSpace` with the group of newly created `PermutationSpace` dictionaries.
3. In `permutationSpace`, do not overwrite or delete a `Leaf` pinned at a `Pile`.

## Data structures for better performance

| Type      | Elements | Ordered | Notes                                      |
| --------- | -------- | ------- | ------------------------------------------ |
| iterator  | fixed?   | no      | Avoid unneeded evaluation. Single-use only |
| frozenset | fixed    | no      |                                            |
| set       | changing | no      |                                            |
| range     | fixed    | yes     | Single-use only                            |
| tuple     | fixed    | yes     |                                            |
| deque     | changing | yes     | Fast at ends; slow index                   |
| list      | changing | yes     |                                            |

## 2^n-dimensional maps

The following observations are due to two factors: the map's geometry and "leveraged" enumeration: we enumerate a subset of the foldings.

Truth:

- leaf0 is always in pile0
- leaf1 is always in pile1
- leaf首零 is always in `state.pileLast`

### In a `Folding`, the absolute value of the difference between adjacent leaves is a value in `state.productsOfDimensions[0:-1]`

Analytically, the leaves differ by one in exactly one dimension, which means the difference between adjacent leaves is always a power of 2.

#### The sum of differences between adjacent leaves equals the difference of the `leaf` in `state.pileLast` and the `leaf` in the first `pile`

- For each `leaf_k` in `pileOfLeaf_k`, `leaf_r` is in `pileOfLeaf_k + 1`, and the difference between `leaf_r` and `leaf_k` is `leaf_r - leaf_k`.
- Retain the sign.
- The sum of the differences is `state.leavesTotal // 2`.

#### The total number of differences in a `Folding` has symmetry

- The total number of differences is `state.leavesTotal - 1`, which is an odd number.
- Reminder: `state.productsOfDimensions[-1] == state.leavesTotal`, and `state.productsOfDimensions[-2] == state.leavesTotal // 2`.
- The difference between two adjacent leaves may be 1, -1, 2, -2, 4, -4, ..., `state.leavesTotal // 2`, `- (state.leavesTotal // 2)`.
- The total number of differences equal to `state.leavesTotal // 2` is always exactly one more than the total number of differences equal to `- (state.leavesTotal // 2)`.
- For all other magnitudes in `state.productsOfDimensions[0:-2]`, the total number of differences of that magnitude is always equal to the total number of differences of the negative of that magnitude.
- The signs of the magnitudes alternate: if the difference between two leaves is 2, for example, then before there can be another difference of 2, there must be a difference of -2.
- Because `state.leavesTotal // 2` always has one more than `- state.leavesTotal // 2`, the first and last differences with magnitude `state.leavesTotal // 2` are positive.

#### The absolute value of the differences in two consecutive piles cannot be the same

At `pile_k`, for example, if the difference is -4, then at `pile_k + 1`, the difference cannot be -4 or 4.

#### The running total of the differences does not repeat in a `Folding`

- The running total is a distinct integer in the range `[0, state.leavesTotal)`.
- At pile0, the running total is 0.
- At pile1, the running total is 1.
- At `state.pileLast`, the running total is `state.leavesTotal // 2`.

### Given leaves `k` and `r`, if `dimensionNearest首(k) <= dimensionNearestTail(r)`, then `pileOf_k < pileOf_r`

Physically, `pileOf_r` can exist before `pileOf_k`, but due to Lunnon Theorem 4, we can enumerate a subset of foldings and multiply by a formula.

`dimensionNearest首(k)` is a 0-based index. Use the index on `state.productsOfDimensions`, and you get the lowest value of `r`. Furthermore, `k` precedes all multiples of `r`. This gives you multiple simple ways to make a list of the values of `r`.

```python
k: int # Is a leaf
index = dimensionNearest首(k)

rTheFirst = state.productsOfDimensions[index]
step = rTheFirst

leavesThatCannotPrecede_k = range(rTheFirst, state.leavesTotal, step)
```

### Pairs of leaves with low entropy

For example, a 2^6-dimensional map has 7840 total sequences that must be enumerated. Two pairs of leaves are always consecutive in a `Folding`, and I call them the "beans and cornbread" pairs. A few other pairs are regularly consecutive: the sets of pairs and their relatively entropy are predictable.

| Consecutive Sequences | 1° leaf | 2° leaf | 1° leaf  | 2° leaf    |
| --------------------- | ------- | ------- | -------- | ---------- |
| 7840                  | 3       | 2       | 一+零    | 一         |
| 7840                  | 16      | 48      | 首一     | 首零一     |
| 6241                  | 5       | 4       | 二+零    | 二         |
| 6241                  | 6       | 7       | 二+一    | 二+一+零   |
| 6241                  | 8       | 40      | 首二     | 首零二     |
| 6241                  | 56      | 24      | 首零一二 | 首一二     |
| 5897                  | 4       | 36      | 二       | 首零二     |
| 5897                  | 9       | 8       | 零+首二  | 首二       |
| 5889                  | 10      | 11      | 一+首二  | 零+一+首二 |
| 5889                  | 52      | 20      | 首零一三 | 首一三     |

Interestingly, the 22 pairs of `leaf二一, leaf二一零` in consecutive piles have a very small combined domain, only 76 pairs, but 22 pairs cover 80% of the 7840 sequences and the other 54 (non-consecutive) pairs only cover 20%. Furthermore, in the 22 consecutive pairs, `leaf二一零` follows `leaf二一`, but in the rest of the domain, `leaf二一` always follows `leaf二一零`.

## Semiotics, notation, and givens

- Each `leaf` is a distinct integer in the range `[0, state.leavesTotal)`.
- Each `pile` is a distinct integer in the range `[0, state.pilesTotal)`.
