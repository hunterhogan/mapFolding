# What I think I know about my "elimination" algorithm

## Rules for maintaining a valid permutation space

1. In `leavesPinned`, if `leaf` is not pinned, deconstruct `leavesPinned` by the `pile` domain of `leaf`.
   1. For each `pile` in the domain of `leaf`, if `pile` in `leavesPinned` is not occupied, create a new `PermutationSpace` dictionary by appending `leaf` pinned at `pile` to `leavesPinned`.
   2. Replace `leavesPinned` with the group of newly created `PermutationSpace` dictionaries.
2. In `leavesPinned`, if a `pile` is not pinned, deconstruct `leavesPinned` by the `leaf` range (mathematical range) of `pile`.
   1. For each `leaf` in the range of `pile`, if `leaf` is not already pinned in `leavesPinned`, create a new `PermutationSpace` dictionary by appending `leaf` pinned at `pile` to `leavesPinned`.
   2. Replace `leavesPinned` with the group of newly created `PermutationSpace` dictionaries.
3. In `leavesPinned`, do not overwrite or delete a `leaf` pinned at a `pile`.

## Data structures for better performance

| Type      | Elements | Ordered | Notes                     |
| --------- | -------- | ------- | ------------------------- |
| iterator  | fixed?   | no      | Avoid unneeded evaluation |
| frozenset | fixed    | no      |                           |
| set       | changing | no      |                           |
| tuple     | fixed    | yes     |                           |
| list      | changing | yes     |                           |

## 2^ⁿ-dimensional maps: given leaves `k` and `r`, if `dimensionNearest首(k) <= dimensionNearestTail(r)`, then `pileOf_k < pileOf_r`

Physically, `pileOf_r` can exist before `pileOf_k`, but due to Lunnon Theorem 4, we can enumerate a subset of foldings and multiply by a formula.

`dimensionNearest首(k)` is a 0-based index. Use the index on `state.productsOfDimensions`, and you get the lowest value of `r`. Furthermore, `k` precedes all multiples of `r`. This gives you multiple simple ways to make a list of the values of `r`.

```python
k: int # Is a leaf
index = dimensionNearest首(k)

rTheFirst = state.productsOfDimensions[index]
step = rTheFirst

leavesThatCannotPrecede_k = range(rTheFirst, state.leavesTotal, step)
```
