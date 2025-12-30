# What I think I know about my "elimination" algorithm

## Rules for maintaining a valid permutation space

1. In `leavesPinned`, if `leaf` is not pinned, deconstruct `leavesPinned` by the `pile` domain of `leaf`.
   1) For each `pile` in the domain of `leaf`, if `pile` in `leavesPinned` is not occupied, create a new `PermutationSpace` dictionary by appending `leaf` pinned at `pile` to `leavesPinned`.
   2) Replace `leavesPinned` with the group of newly created `PermutationSpace` dictionaries.
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

## Algorithm concept: Bureaucratic Permutation Generator

Generating a constrained sequence of permutations.

1. A column is an independent copy of the domain of all possible values.
   1. The column has a distinct index corresponding to its position on a line.
   2. The first column is at position 0 and its index is 0.
   3. Each additional column is immediately adjacent to previous column.
2. A position is a location in an uninterrupted linear sequence positions.
   1. If the position has a corresponding column hold, the position may or may not have a value assigned from the domain of the column.
   2. Position and column indices increase by one with each move to the right.
3. The carriage starts at position -1, which does not have a column, and moves to the right, which is position 0.
   1. The scribe performs an action, and the carriage moves.
      1. The possible scribe actions are assign a value, assign no value, or collect values.
      2. The possible carriage moves are move to the right or move to the left.
   2. The scribe checks if the position does or does not have a column.
      1. If the position has a column, the scribe checks if the domain of the column does or does not have values.
         1. If the domain of the column has values, the scribe assigns a value from the domain of the column. The carriage moves to the right.
         2. If the domain of the column does not have values, the scribe assigns no value. The carriage moves to the left.
      2. If the position does not have a column, the scribe checks if the positions to the left have values or are empty.
         1. If the positions to the left have values, the scribe records the sequence of values in the list of sequences. The carriage moves to the left.
         2. If the positions to the left are empty, the scribe does nothing (because the carriage has moved to position -1). The carriage does not move.
4. A filter reduces the domain of possible values for one or more columns.
   1. The scope of the filter defines the columns that are affected by the filter.
   2. An expiration index of a filter defines when the filter permanently stops affecting the columns. (See janitor.)
   3. When a filter is created, an unchangeable expiration index corresponding to the position to the left of the carriage is automatically added to the filter.
   4. A filter may have more than one expiration index.
   5. An expiration index must not be in the scope of columns affected by the filter.
   6. The filter must be independent; the filter application order must be irrelevant.
5. When the carriage moves to a position, the janitor¹ deletes each filter that has an expiration index matching the index of the carriage's position and removes the column's assigned value.
6. A bureaucrat has exactly one set of conditions it compares to the state of the system, and if the conditions match the state of the system, the bureaucrat creates exactly one filter.
   1. The system has standard bureaucrats.
      1. For each column, one bureaucrat is automatically hired to monitor the domain of the column.
         1. Conditions to match: the domain of the column is empty, and if the column does not have an assigned value, the bureaucrat creates a filter.
         2. The filter's effect: remove all values from the column's domain. The scope: all columns to the right of the carriage.
      2. Each column has a bureaucrat that monitors the assigned value of the column.
         1. Condition: if the column has an assigned value, the bureaucrat creates a filter.
         2. The filter's effect: remove the assigned value from a column's domain. The scope: the bureaucrat's column and all columns to the right of the column.
   2. To restrict a column's value to exactly one possibility, for example, implement a bureaucrat to create a filter before the first move that restricts the domain of the column to the desired value.
7. Steps occur in the following order and do not overlap with other steps:
   1. Bureaucrats create filters.
   2. The carriage moves.
   3. The janitor acts.
   4. The scribe acts.

Technical details

- The bureaucrats can and probably should work concurrently with each other.
- This is not a recursive or backtracking permutation generator. Compare with, for example, [TheAlgorithms](https://github.com/TheAlgorithms/Python/blob/master/data_structures/arrays/permutations.py).

Observations

- This complex system relies on simple rules and actions.
- The "actors" should be stupid, weak, and ignorant. Note that the scribe, the most sophisticated actor, for example, doesn't know the position index, the most important data point.
- Implement your constraints by "hiring" a bureaucrat to create a filter.
- The janitor is not a bureaucrat.
- Filters only restrict the domain.
- Nothing can add to the domain.
- The actors are not part of the state of the system.
- The filters are not part of the state of the system.
- After their creation, filters cannot be modified.
- Filters cannot be deleted: they expire automatically.
- Create concurrency by dividing the domain of one column into non-overlapping subdomains and running multiple instances of the system, each with one subdomain.

Question: is the state equal to: the assigned values and the domains of the columns?

¹ "[Janitor](https://www.etymonline.com/word/janitor)" is closely linked to "Janus", the Roman "guardian god of portals, doors, and gates."
