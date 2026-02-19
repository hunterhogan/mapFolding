# What I think I know about my "elimination" algorithm

**Important**: The last section of this document is a "glossary" with critical contextual details. You ought to skim it now and consult it as needed.

## Rules for maintaining a valid permutation space

1. In a dictionary of `Pile` keys and `LeafSpace` values, if `leaf` is not pinned, deconstruct `permutationSpace` by the `Pile` domain of `leaf`.
   1. For each `pile` in the domain of `leaf`, if `pile` in `permutationSpace` is not occupied, create a new `PermutationSpace` dictionary by appending `leaf` pinned at `pile` to `permutationSpace`.
   2. Replace `permutationSpace` with the group of newly created `PermutationSpace` dictionaries.
2. In a `PermutationSpace`, if a `pile` is not pinned, deconstruct `permutationSpace` by the `Leaf` range (mathematical range) of `pile`.
   1. For each `leaf` in the range of `pile`, if `leaf` is not already pinned in `permutationSpace`, create a new `PermutationSpace` dictionary by appending `leaf` pinned at `pile` to `permutationSpace`.
   2. Replace `permutationSpace` with the group of newly created `PermutationSpace` dictionaries.
3. In `permutationSpace`, do not overwrite or delete a `Leaf` pinned at a `Pile`.

## Data structures for better performance

| Type        | Stable   | Ordered | Best For...       | Performance / Limitation Note                           |
| :---------- | :------- | :------ | :---------------- | :------------------------------------------------------ |
| iterator    | Fixed    | Yes     | Lazy processing   | Memory-efficient; single-use only.                      |
| range       | Fixed    | Yes     | Integer sequences | $O(1)$ memory; $O(1)$ membership check.                 |
| frozenset   | Fixed    | No      | Membership keys   | Hashable; used as keys for other sets/dicts.            |
| tuple       | Fixed    | Yes     | Static records    | Lower memory overhead than lists; faster iteration.     |
| NamedTuple  | Fixed    | Yes     | Named records     | Tuple performance with object-like access.              |
| set         | Changing | No      | Uniqueness        | $O(1)$ lookup; high memory overhead (\~32 bytes/item).  |
| Counter     | Changing | Yes     | Tallying          | Specialized for frequencies; supports multiset math.    |
| deque       | Changing | Yes     | Stacks / Queues   | $O(1)$ at ends; $O(n)$ in middle; thread-safe ends.     |
| list        | Changing | Yes     | General use       | $O(1)$ index; $O(n)$ insert/delete at start.            |
| array.array | Changing | Yes     | Numeric data      | Stores raw C-types; memory compact; better cache use.   |
| SortedList  | Changing | Sort    | Searchable data   | Maintains order automatically; $O(\\log n)$ operations. |

## 2^n-dimensional maps

The following observations are due to two factors: the map's geometry and "leveraged" enumeration: we enumerate a subset of the foldings.

Truth:

1. `leafOrigin` (leaf0) is always in `pileOrigin` (pile0)
2. `leaf1` is always in `pile1`
3. `leaf首零` is always in `state.pileLast`

### Given a `Folding` with `leaf` adjacent to `leaf_r`, the difference of `leaf` and `leaf_r` is a power of 2, with many restrictions

Reminders and general observations:

- [something, something] absolute value [yada yada] in `state.productsOfDimensions[0:-1]`
- The difference between two adjacent leaves may be 1, -1, 2, -2, 4, -4, ..., `pos(state.leavesTotal // 2)`, `neg(state.leavesTotal // 2)`.
- The total number of differences is `state.leavesTotal - 1`, which is an odd number. (_Cf._ fencepost problem.)
- `state.productsOfDimensions[-1] == state.leavesTotal`, and `state.productsOfDimensions[-2] == state.leavesTotal // 2`.

Differences:

1. The signs of the magnitudes alternate: if the difference between two leaves is +2, for example, then before there can be another difference of +2, there must be a difference of -2.
2. The total number of differences equal to `pos(state.leavesTotal // 2)` is always exactly one more than the total number of differences equal to `neg(state.leavesTotal // 2)`.
   1. Therefore, the first and last differences with magnitude `state.leavesTotal // 2` are positive.
3. For all other magnitudes in `state.productsOfDimensions[0:-2]`, the total number of positive and negative differences is always equal.
   1. Therefore, the first and last differences with those magnitudes must have opposite signs.
   2. Given Truth 1 and Truth 2,
      1. every `Folding` has at least one difference of +1 and one difference of -1,
      2. the first difference is +1, and
      3. the last difference of magnitude 2^0 (1) is -1.
4. In two consecutive piles, the absolute value of the differences cannot be the same. Given the difference at `pile_k` is -4, for example, then at `pile_k + 1`, the difference cannot be -4 or +4.
5. The sum of all differences is `state.leavesTotal // 2`.
6. Starting from `pileOrigin` in a `Folding`, the running total of differences is a distinct integer in the range `[0, state.leavesTotal)` and does not repeat.

### Given leaves `k` and `r`, if `dimensionNearest首(k) <= dimensionNearestTail(r)`, then `pileOf_k < pileOf_r`

Physically, `pileOf_r` can exist before `pileOf_k`, so the limitation is due to leveraged enumeration.

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

| Sequences | 1° leaf | 2° leaf | 1° leaf  | 2° leaf    |
| --------- | ------- | ------- | -------- | ---------- |
| 7840      | 3       | 2       | 一+零    | 一         |
| 7840      | 16      | 48      | 首一     | 首零一     |
| 6241      | 5       | 4       | 二+零    | 二         |
| 6241      | 6       | 7       | 二+一    | 二+一+零   |
| 6241      | 8       | 40      | 首二     | 首零二     |
| 6241      | 56      | 24      | 首零一二 | 首一二     |
| 5897      | 4       | 36      | 二       | 首零二     |
| 5897      | 9       | 8       | 零+首二  | 首二       |
| 5889      | 10      | 11      | 一+首二  | 零+一+首二 |
| 5889      | 52      | 20      | 首零一三 | 首一三     |

Interestingly, the 22 pairs of `leaf二一, leaf二一零` in consecutive piles have a very small combined domain, only 76 pairs, but 22 pairs cover 80% of the 7840 sequences and the other 54 (non-consecutive) pairs only cover 20%. Furthermore, in the 22 consecutive pairs, `leaf二一零` follows `leaf二一`, but in the rest of the domain, `leaf二一` always follows `leaf二一零`.

Furthermore, it is easy to predict the pairs of pairs-of-leaves with low entropy.

1. (一+零, 一) and (首一, 首零一)
2. (leaf二一, leaf二一零) and (leaf二零, leaf二)
3. (leaf首二, leaf首零二) and (leaf首零一二, leaf首一二)

## Semiotics, notation, and givens

- Each `Leaf` is a distinct integer in the range `[0, state.leavesTotal)`.
- `leaf` is the archetypal variable name for a `Leaf`.
- Each `Pile` is a distinct integer in the range `[0, state.pilesTotal)`.
- `pile` is the archetypal variable name for a `Pile`.
- A `Folding` is a one-to-one correspondence between the set of `Pile` and the set of `Leaf`.
- `folding` is the archetypal variable name for a `Folding`.
****- A `PermutationSpace` is an exclusive subset of the undifferentiated permutation space of the factorial of `state.leavesTotal`.
