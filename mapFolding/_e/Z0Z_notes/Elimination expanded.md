# What I think I know about my "elimination" algorithm

- [What I think I know about my "elimination" algorithm](#what-i-think-i-know-about-my-elimination-algorithm)
	- [Rules for maintaining a valid permutation space](#rules-for-maintaining-a-valid-permutation-space)
	- [Data structures for better performance](#data-structures-for-better-performance)
	- [2^n-dimensional maps](#2n-dimensional-maps)
		- [Given a `Folding` with `leaf` adjacent to `leaf_r`, the difference of `leaf` and `leaf_r` is a power of 2, with many restrictions](#given-a-folding-with-leaf-adjacent-to-leaf_r-the-difference-of-leaf-and-leaf_r-is-a-power-of-2-with-many-restrictions)
		- [Given leaves `k` and `r`, if `dimensionNearest首(k) <= dimensionNearestTail(r)`, then `pileOf_k < pileOf_r`](#given-leaves-k-and-r-if-dimensionnearest首k--dimensionnearesttailr-then-pileof_k--pileof_r)
		- [Pairs of leaves with low entropy](#pairs-of-leaves-with-low-entropy)
		- [Crease neighbors are `bit_flip` neighbors](#crease-neighbors-are-bit_flip-neighbors)
		- [Addends for "next" and "prior" leaves match crease neighbors](#addends-for-next-and-prior-leaves-match-crease-neighbors)
		- [Progressions within a dimension](#progressions-within-a-dimension)
		- [Bit inversion symmetry](#bit-inversion-symmetry)
		- [Leaf domains are directly tied to `sumsOfProductsOfDimensions` and `sumsOfProductsOfDimensionsNearest首`](#leaf-domains-are-directly-tied-to-sumsofproductsofdimensions-and-sumsofproductsofdimensionsnearest首)
		- [Leaf precedence hierarchy](#leaf-precedence-hierarchy)
		- [Pile-range formulas: relationship between pile ranges and leaf domains](#pile-range-formulas-relationship-between-pile-ranges-and-leaf-domains)
		- [Constraint propagation system](#constraint-propagation-system)
		- [Domain-based exclusions at specific piles](#domain-based-exclusions-at-specific-piles)
		- [Divisibility and Theorem multipliers](#divisibility-and-theorem-multipliers)
		- [Low-entropy leaf pairs: structure of "beans and cornbread"](#low-entropy-leaf-pairs-structure-of-beans-and-cornbread)
		- [Pinning pile `(2,6)` — differences and addend lists](#pinning-pile-26--differences-and-addend-lists)
		- [General observations for analyzing pinning](#general-observations-for-analyzing-pinning)
		- [Leaf metadata per dimension](#leaf-metadata-per-dimension)
		- [Forbidden inequalities (IFF checking)](#forbidden-inequalities-iff-checking)
	- [Semiotics, notation, and givens](#semiotics-notation-and-givens)


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
      1. the first difference in every `Folding` is +1,
      2. the last difference of magnitude 1 is -1.
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

### Crease neighbors are `bit_flip` neighbors

For $2^n$-dimensional maps, the crease neighbors of a `leaf` are obtained by flipping individual bits of the `leaf`'s binary representation. That is, the "next" and "prior" crease neighbors of `leaf` along `dimension` are `bit_flip(leaf, dimension)`. But not all bit-flips are valid creases: the crease list is a strict subset of the full list of bit-flips, and which bit-flips survive depends on the leaf's parity structure.

Specifically, `getLeavesCreasePost(state, leaf)` and `getLeavesCreaseAnte(state, leaf)` each return a subset of `[bit_flip(leaf, d) for d in range(dimensionsTotal)]`. The subsets are selected by slicing indices determined by `howManyDimensionsHaveOddParity(leaf)`, `dimensionNearestTail(leaf)`, and `dimensionNearest首(leaf)`.

- If the parity of `leaf` is even, then all crease-post neighbors are greater than `leaf`, and all crease-ante neighbors are less than `leaf`.
- If the parity of `leaf` is odd, then all crease-post neighbors are less than `leaf`, and all crease-ante neighbors are greater than `leaf`.

Furthermore, the number of crease-ante neighbors of a leaf plus one equals the number of crease-post neighbors of that leaf:

$$|\text{crease-ante}(\text{leaf})| + 1 = |\text{crease-post}(\text{leaf})|$$

This is verified empirically in the hypothesis code.

### Addends for "next" and "prior" leaves match crease neighbors

The addends that produce the next-in-sequence and prior-in-sequence leaves from a given `leaf` are exactly the crease-post and crease-ante neighbors (respectively), obtained via `bit_flip`. In other words:

```python
listAddendLeaves == listLeavesNext        # True for all leaves
listAddendPriorLeaves == listLeavesPrior  # True for all leaves
```

where `listLeavesNext` and `listLeavesPrior` are slices of `[bit_flip(leaf, dimension) for dimension in range(dimensionsTotal)]` selected by the same slicing logic as `getLeavesCreasePost` and `getLeavesCreaseAnte`.

### Progressions within a dimension

Consider a dimension origin $d$ (a power of 2, e.g., 16). The leaf $d$ has trailing zeros in binary equal to $\log_2(d)$. Each trailing zero is a "sub-dimension origin" relative to $d$. Starting from each sub-dimension origin and accumulating the sums of products of dimensions produces the odd leaves in the range $[d, 2d)$:

```text
sums: 0, 1, 3, 7, 15, ...
For dimension origin 16:
cf: 16, 17, 19, 23, 31
cf: 18, 19, 21, 25, ...
cf: 20, 21, 23, 27, ...
```

Counting from the end (applying `bit_mask(dimensionsTotal) ^ leaf`) yields the even leaves. This "counting from the end" is a permutation expressible by bit manipulation: for each leaf $l$ in the first half, $l \oplus \text{bit\_mask}(n)$ gives the corresponding leaf in the second half.

### Bit inversion symmetry

For a $2^n$-dimensional map with $\text{leavesTotal} = 2^n$ leaves, the pile-range of `pile = leavesTotal // 2 - 1` has a two-part structure:

1. Leaves in $[0, \text{leavesTotal}//2)$.
2. Their bit-inversions in $[\text{leavesTotal}//2, \text{leavesTotal})$.

The bit inversion is $\text{leaf} \oplus \text{bit\_mask}(n)$:

```python
for leaf in [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31]:
    inverted = leaf ^ bit_mask(6)
# 31 -> 32, 26 -> 37, 25 -> 38, 22 -> 41, 21 -> 42, ...
```

This extends: `leafInSubHyperplane` can be used for the "start over" equivalences at $\text{leavesTotal}/4$, $\text{leavesTotal}/2$, and $3\cdot\text{leavesTotal}/4$.

### Leaf domains are directly tied to `sumsOfProductsOfDimensions` and `sumsOfProductsOfDimensionsNearest首`

For dimension origins (powers of 2), leaf domains follow a pattern:

- The `start` of a dimension origin's domain equals `sumsOfProductsOfDimensions` up to that dimension.
- The `stop` equals `sumsOfProductsOfDimensionsNearest首` from the head minus an offset of 2.

For a $2^6$-dimensional map:

| Dimension origin | Domain start | Domain stop |
| :--------------- | :----------- | :---------- |
| 1                | 1            | 2           |
| 2                | 3            | 34          |
| 4                | 7            | 50          |
| 8                | 15           | 58          |
| 16               | 31           | 62          |
| 32               | 63           | 64          |

The full formula for `getLeafDomain`:

```python
range(
    sumsOfProductsOfDimensions[dimensionNearestTail(leaf) + inclusive]
        + howManyDimensionsHaveOddParity(leaf) - originPinned
    , sumsOfProductsOfDimensionsNearest首[dimensionNearest首(leaf)]
        + 2 - howManyDimensionsHaveOddParity(leaf) - originPinned
    , 2 + (2 * (leaf == 首零(dimensionsTotal) + 零))
)
```

### Leaf precedence hierarchy

A hierarchy of facts; each statement is _necessarily_ true about statements below it. If two statements appear to contradict each other, apply the superior statement to its full scope, and apply the inferior statement only where it does not contradict the superior statement.

1. `leafOrigin` precedes all other leaves.
2. `leaf零` precedes all other leaves except `leafOrigin`.
3. `leaf首零` is preceded by all other leaves.

Some leaves are always preceded by one or more leaves. Most leaves, however, are preceded by one or more other leaves only if the leaf is in a specific pile. These conditional predecessor relationships are computed by `getDictionaryConditionalLeafPredecessors`.

### Pile-range formulas: relationship between pile ranges and leaf domains

The leaves in the pile-range of a pile can be computed from leaf domains (which leaf can appear at which pile). For odd piles in a specific range (e.g., $9 \le \text{odd piles} \le 47$ for a $2^6$-dimensional map), the pile-range follows a pattern involving `intraDimensionalLeaves` and `productsOfDimensions`, though a full closed-form formula remains under development.

The pile-range construction involves three groups:

1. **Odd leaves below $\text{leavesTotal}/2$** and some even leaves, via `sumsOfProductsOfDimensions` addends scaled by `productsOfDimensions`.
2. **Even leaves above $\text{leavesTotal}/2$**, via bit-inversion (`invertLeafIn2上nDimensions`) of the first group's formula.
3. **Dimension origins** and their inverses, added separately.

### Constraint propagation system

The `pin2上nDimensionalAnnex` module implements a constraint-satisfaction system expressed as multiple specialized reduction functions:

1. **Crease adjacency**: enforce that crease neighbors respect pile ordering.
2. **Pinned leaf propagation**: when a leaf is pinned at a pile, remove it from all other piles.
3. **Head-before-tail ordering**: enforce $\text{pile}(\text{head leaf}) < \text{pile}(\text{tail leaf})$ ordering.
4. **Conditional predecessors**: enforce pile-dependent predecessor constraints.
5. **Crossed crease detection**: eliminate permutation spaces where two creases cross (Koehler 1968 / Legendre 2014 forbidden inequalities).
6. **Non-consecutive dimensions**: in two consecutive piles, the absolute value of the difference cannot repeat.
7. **Domain size one**: when a pile's `LeafOptions` reduces to a single leaf, pin it.
8. **Naked subset elimination**: when $k$ piles share the same $k$ candidate leaves, eliminate those leaves from all other piles.

These functions are interdependent components, not independent algorithms. Each assumes the others will run afterward to propagate consequences.

### Domain-based exclusions at specific piles

For $2^n$-dimensional maps, certain leaves can be excluded from `pile 零Ante首零` (and similar piles near the ends) based on which leaves are pinned at the first four "orders" of piles ($\text{pile} \le 4$ or $\text{pile} \ge 首 - 4$). This is encoded in `pinPile零Ante首零AfterDepth4`.

Key observations:

- A leaf in `pile一零` does not itself have a crease neighbor in the pile-range of `pile零Ante首零`, but `leafInSubHyperplane(leafAt一零)` _does_. The `ptount` function leverages this sub-hyperplane projection.
- The same pattern applies to `pile首Less一零`: the leaf at that pile doesn't have a crease neighbor in range, but its sub-hyperplane projection does.
- All single-leaf-based exclusions are resolved for the first four orders. Beyond that, multi-leaf interactions ("knock-out" leaves) become more complex.

### Divisibility and Theorem multipliers

For $2^n$-dimensional maps:

- `foldsTotal` is divisible by $\text{leavesTotal} \times 2^{\text{dimensionsTotal}} \times \text{dimensionsTotal}!$
- This includes: `Theorem2aMultiplier = leavesTotal` (pin `leafOrigin` at `pileOrigin`), and `Theorem4Multiplier = dimensionsTotal!` (factorial symmetry from permuting dimensions of equal length).
- For general maps with unequal dimension lengths, `Theorem2Multiplier = 2` when the longest dimension has length $> 2$ and $\text{leavesTotal} > 4$, and `Theorem4Multiplier` is the product of factorials of groups of equal-length dimensions.

### Low-entropy leaf pairs: structure of "beans and cornbread"

The "beans and cornbread" pattern refers to pairs of leaves that are _always_ consecutive in every enumerated `Folding`. Two such pairs exist in every $2^n$-dimensional map:

1. `(leaf一零, leaf一)` and `(leaf首一, leaf首零一)` — dimension-一 pairs.
2. These pairs constrain domain functions: `getDomainDimension一` returns 4-tuples of piles for all four of these leaves jointly.

Beyond the universal pairs, additional pairs have low entropy. In a $2^6$-dimensional map, the pair `(leaf二一, leaf二一零)` appears in consecutive piles in 22 out of 76 pile-pairs, but those 22 consecutive pairs cover 80% of the 7840 enumerated sequences. In the consecutive case, `leaf二一零` follows `leaf二一`; in the remaining domain, the order reverses.

Predictable pairs of low-entropy pairs follow a "mirroring" pattern:

1. `(一+零, 一)` and `(首一, 首零一)` — "mirror" across the map.
2. `(leaf二一, leaf二一零)` and `(leaf二零, leaf二)`.
3. `(leaf首二, leaf首零二)` and `(leaf首零一二, leaf首一二)`.

### Pinning pile `(2,6)` — differences and addend lists

For a $2^6$-dimensional map investigated empirically, each leaf has a set of "addends" that produce its next-in-sequence and prior-in-sequence neighbors. These addend lists have specific structures: they start with a step that depends on the leaf's dimension-origin membership, and grow by appending products-of-dimensions terms.

For leaf-pairs $(l, r)$ in consecutive piles, the "satellite" leaves (other leaves whose piles are constrained by the pair) can be computed:

```python
# For leaf-pair (l, r) with l > r:
ll = [l + cumulative_sum(listLeft[0:i]) for i ...]
rr = [r + cumulative_sum(listRight[0:i]) for i ...]
# With the "larger" leaf getting one extra addend term at the tail.
```

Some leaf-pairs have unexplained anomalies (additional satellites reachable by $+4$ offsets, or specific "knock-out" relationships with other pairs, e.g., `(2,6)` is on the list of `(3,2)`; `(24,16)` is on the list of `(16,48)`).

### General observations for analyzing pinning

For a $2^6$-dimensional map with 7840 enumerated sequences:

- Piles 2 and 62 are the first variable piles, each having only 5 possible leaf assignments.
- The permutations of piles $\{2, 16, 32, 48, 62\}$ produce 5730 of 7840 sequences.
- The number of distinct leaf possibilities at these piles: 5, 29, 30, 29, 5.

Maps of shape $(3, 3, \ldots, 3)$ have `foldsTotal` divisible by $\text{leavesTotal} \times 2^{\text{dimensionsTotal}} \times \text{dimensionsTotal}!$

### Leaf metadata per dimension

For each leaf, per dimension:

- **For inequality checking**: the next crease leaf (or `None`), and the parity in that dimension.
- **Domain of leaf**: the range of piles where the leaf may appear.
- **Range of leaves in piles**: the set of leaves that may appear at a given pile.

### Forbidden inequalities (IFF checking)

To confirm a multidimensional folding is valid, confirm that each constituent one-dimensional section is valid. To confirm a one-dimensional section is valid, check that no two creases cross.

A "crease" is shorthand for two leaves that are physically connected. In a one-dimensional section, each leaf connects to at most two neighbors (before and after). Two creases $(k, k{+}1)$ and $(r, r{+}1)$ with matching parity ($k \equiv r \pmod{2}$) cross when their pile positions satisfy any of four forbidden orderings (Legendre 2014, simplifying Koehler 1968's eight):

$$[\pi(k) < \pi(r) < \pi(k{+}1) < \pi(r{+}1)] \quad [\pi(k{+}1) < \pi(r{+}1) < \pi(k) < \pi(r)]$$
$$[\pi(r{+}1) < \pi(k) < \pi(r) < \pi(k{+}1)] \quad [\pi(k) < \pi(r{+}1) < \pi(k{+}1) < \pi(r)]$$

Lunnon's theorem: a pile ordering is a valid folding if and only if all its one-dimensional sections are valid (non-crease-crossing).

## Semiotics, notation, and givens

- Each `Leaf` is a distinct integer in the range `[0, state.leavesTotal)`.
- `leaf` is the archetypal variable name for a `Leaf`.
- Each `Pile` is a distinct integer in the range `[0, state.pilesTotal)`.
- `pile` is the archetypal variable name for a `Pile`.
- A `Folding` is a one-to-one correspondence between the set of `Pile` and the set of `Leaf`.
- `folding` is the archetypal variable name for a `Folding`.
- A `PermutationSpace` is an exclusive subset of the undifferentiated permutation space of the factorial of `state.leavesTotal`.
- The positional-numeral ideographs (零, 一, 二, 三, etc.) represent `productsOfDimensions[dimensionIndex]`, i.e., powers of 2 in $2^n$-dimensional maps: 零 $= 2^0 = 1$, 一 $= 2^1 = 2$, 二 $= 2^2 = 4$, etc.
- 首 (shǒu, "head") denotes the most-significant-dimension end of a coordinate. For example, `首零(n)` $= 2^{n-1}$, `首一(n)` $= 2^{n-2}$, `首零一(n)` $= 2^{n-1} + 2^{n-2}$.
- `LeafOptions` is a `gmpy2.mpz` bitset where bit $i$ is set iff leaf $i$ is a candidate at that pile.
- `PinnedLeaves` is a `dict[Pile, Leaf]` of pile-to-leaf assignments that are determined.
- `UndeterminedPiles` is a `dict[Pile, LeafOptions]` of pile-to-candidate-set mappings still to be resolved.
