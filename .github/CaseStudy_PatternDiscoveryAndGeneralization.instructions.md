# Case Study: Pattern Discovery and Generalization in Combinatorics Code

## Audience and Purpose

This document is for AI assistants (including future versions of myself) working on the mapFolding project. The objective is to demonstrate **how** to deconstruct complex combinatorial data into simple component rules, then combine those rules into code that:

1. Reproduces the original static data exactly
2. Scales to maps with more dimensions than the original dataset

## The Problem Domain

The mapFolding project deals with multidimensional map folding algorithms. Maps are hyperplanes with `dimensionsTotal` dimensions, each with length 2, giving `2^dimensionsTotal` total "leaves" (positions). The challenge is discovering which leaves must precede others at specific "piles" (positions in a folding sequence).

## Concrete Example: `dictionary首零Plus零` in `_dataDynamic.py`

### The Starting Point: Static Data

The process began with tuples extracted from empirical analysis of a 6-dimensional map (`dimensionsTotal = 6`, `leavesTotal = 64`). Each tuple `(leafPredecessor, pileFirst)` means: "at piles starting from `pileFirst`, the leaf `leafPredecessor` must come before `leaf首零Plus零`."

```python
leafPredecessorPileFirst = [
    ( 2, 6), ( 3, 6),
    (34, 6), (35, 6),
    ( 6, 34), ( 7, 34),
    (38, 34), (39, 34),
    ( 4, 38), ( 5, 38),
    (36, 38), (37, 38),
    # ... ~60 more tuples ...
]
```

### Step 1: Physical Rearrangement for Visual Patterns

The first technique was **physically rearranging tuples** into groupings that revealed patterns. This is a human visual technique that may have limited direct applicability to AI, but the principle is: **organize data by suspected shared properties**.

Initial grouping by `pileFirst` value:

```python
(10, 50), (11, 50), (12, 50), (13, 50), (30, 50), (31, 50),
(42, 50), (43, 50), (44, 50), (45, 50), (62, 50), (63, 50),
( 8, 54), ( 9, 54), (22, 54), (23, 54), (26, 54), (27, 54), (28, 54), (29, 54),
(40, 54), (41, 54), (54, 54), (55, 54), (58, 54), (59, 54), (60, 54), (61, 54),
```

This grouping revealed that `pileFirst` values clustered, but the `leafPredecessor` values within each cluster seemed chaotic.

### Step 2: Discovering "Series" Structure

After multiple reorganizations, a "series" structure emerged. The key insight came from analyzing **binary representations**:

```python
#-------- New series ------
# >>> 60^0b111111 = 3
#   0b000011
( 2, 6), ( 3, 6),
(34, 6), (35, 6),     # 0 + 6

#-------- New series ------
# >>> 56^0b111111 = 7
#   0b000111
( 6, 34), ( 7, 34),
(38, 34), (39, 34),
#  0b000101
( 4, 38), ( 5, 38),
(36, 38), (37, 38),     # 32 + 6
```

**Key observations:**

1. Data naturally grouped into "series"
2. Each series had a relationship to powers of 2
3. Within a series, `pileFirst` increased when `howManyDimensionsHaveOddParity` decreased

### Step 3: Iterative Comment Annotation

Comments evolved through iterations:

**Version 1:** `? 8 - 2, 40 - 2, 56 - 2, ? 64 - 2`
**Version 2 (below):** `8 + 32 = 40 + 16 = 56 + 8 = 64`
**Version 3:** `0 + 6, 32 + 6, 48 + 6, 56 + 6`

The progression from arithmetic relationships (`8 + 32 = 40`) to structural relationships (`0 + 6, 32 + 6, ...`) was crucial. The final form matched a known sequence.

### Step 4: Connecting to Existing Functions

The comment `8 + 32 = 40 + 16 = 56 + 8 = 64` triggered recognition of `Z0Z_sumsOfProductsOfDimensionsNearest首`:

```python
magicalSequence = Z0Z_sumsOfProductsOfDimensionsNearest首(state, state.dimensionsTotal)
# Returns: (0, 32, 48, 56, 60, 62, 63)
```

**Critical technique:** The author explicitly catalogs and iterates through existing functions, testing each against the data to expose patterns. This is emphasized as the most valuable technique:

> "I literally went through functions, such as those in `_measure.py`, and tried to use them to expose patterns in the data. I cannot over-emphasize how useful this technique is."

### Step 5: Building the Generalized Formula

With the series structure understood and the magical sequence identified, formulas emerged:

```python
# From comments that captured the discovered rules:
# pileFirst the LAST = magicalSequence + 6
# pileFirst the LAST ascending as leafPredecessor ascends
# Note: howManyDimensionsHaveOddParity(leafPredecessor the LAST) = 1
# pileFirst the others = pileFirst the LAST - 4 * (howManyDimensionsHaveOddParity(...) - 1)
# leafPredecessor the first = Z0Z_invert(state, magicalSequence)
```

### The Final Generalized Code

```python
pileStepAbsolute = 4
for indexUniversal in range(state.dimensionsTotal - 2):
    leafPredecessorTheFirst = Z0Z_invert(state, magicalSequence[state.dimensionsTotal - 2 - indexUniversal])
    leafPredecessorsInThisSeries = state.productsOfDimensions[howManyDimensionsHaveOddParity(leafPredecessorTheFirst)]
    for addend in range(leafPredecessorsInThisSeries):
        leafPredecessor = leafPredecessorTheFirst + (addend * decreasing)
        leafPredecessor首零 = leafPredecessor + 首零(state.dimensionsTotal)
        pileFirst = magicalSequence[indexUniversal] + 6 - (pileStepAbsolute * (howManyDimensionsHaveOddParity(leafPredecessor) - 1 + is_even(leafPredecessor)))
        for pile in listOfPiles[listOfPiles.index(pileFirst): None]:
            dictionary首零Plus零[pile].append(leafPredecessor)
            dictionary首零Plus零[pile].append(leafPredecessor首零)
```

## The Function Catalog: Your Primary Tool

The `mapFolding/_e/` directory contains measurement and semantic functions essential for pattern discovery. Here are the key ones used in this example:

### From `_measure.py`

| Function                                              | Purpose                                                                |
| ----------------------------------------------------- | ---------------------------------------------------------------------- |
| `dimensionNearest首(n)`                               | Position of most significant set bit (0-indexed)                       |
| `dimensionSecondNearest首(n)`                         | Position of second most significant set bit                            |
| `howMany0coordinatesAtTail(n)`                        | Count trailing zeros (CTZ) — how many times n is divisible by 2        |
| `howManyDimensionsHaveOddParity(n)`                   | `bit_count() - 1` — number of set bits excluding the MSB               |
| `leafInSubHyperplane(n)`                              | Project leaf onto sub-hyperplane (bits below MSB)                      |
| `Z0Z_invert(state, n)`                                | XOR with `bit_mask(dimensionsTotal)` — "mirror" within dimension space |
| `Z0Z_sumsOfProductsOfDimensionsNearest首(state, dim)` | Cumulative sums of products of dimensions in reverse                   |

### From `_semiotics.py` (via `_e/__init__.py`)

| Symbol                    | Meaning                                                   |
| ------------------------- | --------------------------------------------------------- |
| `leafOrigin`              | The origin leaf (0)                                       |
| `pileOrigin`              | The origin pile (0)                                       |
| `零, 一, 二, 三, 四, ...` | Dimension indices as powers of 2: 1, 2, 4, 8, 16, ...     |
| `首零(d)`, `首一(d)`, ... | Functions returning leaf indices based on dimension count |
| `decreasing`, `inclusive` | Semantic constants: -1 and 1 respectively                 |

## Methodology: The Pattern Discovery Process

### Phase 1: Data Extraction

1. Generate or extract static data from a specific dimension count (e.g., 6D)
2. Format as tuples or simple structures
3. Verify the data is correct by testing against known foldings

### Phase 2: Physical/Logical Reorganization

1. Group by one property (e.g., `pileFirst`)
2. Sort within groups by another property
3. Look for regularities in group sizes
4. Try multiple grouping strategies

### Phase 3: Binary Analysis

1. Convert values to binary representations
2. Look for bit patterns: trailing zeros, bit counts, MSB positions
3. Test XOR, AND, OR relationships between related values

### Phase 4: Function Catalog Testing

1. **Systematically apply every function in `_measure.py`** to the data values
2. For each function, check if:
   - Output values cluster with observed groupings
   - Output differences match observed step sizes
   - Output equality correlates with tuple membership in same group
3. Document any correlations, even partial ones

### Phase 5: Sequence Recognition

1. Look for known sequences: powers of 2, cumulative sums, products
2. Test against `productsOfDimensions`, `sumsOfProductsOfDimensions`
3. Check if relationships involve dimension-specific offsets

### Phase 6: Formula Construction

1. Start with the simplest relationship that covers the most data
2. Add correction terms for exceptions
3. Express corrections using existing semantic functions
4. Verify formula reproduces ALL original tuples

### Phase 7: Generalization Verification

1. Test on a DIFFERENT dimension count (e.g., 5D or 7D if you have the data)
2. Check edge cases: minimum and maximum leaf values
3. Confirm loop bounds scale with `dimensionsTotal`

## Common Patterns in This Codebase

### The "Series" Pattern

Data often organizes into series indexed by dimension, where:

- Series boundaries relate to `productsOfDimensions` or cumulative sums
- Values within a series relate by `首零(dimensionsTotal)` offsets (adds 2^(d-1))

### The Parity Pattern

`howManyDimensionsHaveOddParity` frequently determines:

- Step sizes within ranges
- Offsets from base values
- The number of items in a group

### The Inversion Pattern

`Z0Z_invert(state, x)` = `x ^ bit_mask(dimensionsTotal)` creates symmetric leaf pairs:

- Leaf `x` and leaf `invert(x)` often have related rules
- Rules for `leaf < 首零` often mirror rules for `leaf >= 首零`

### The Projection Pattern

`leafInSubHyperplane(leaf)` projects to lower dimensions:

- Rules discovered in 6D often apply to the "sub-hyperplane" of the same leaf in 7D
- Use this to bootstrap from smaller dimension analysis

## Warnings and Pitfalls

### "I can see regularity, but I can't find the patterns"

This is normal. The author notes:
> "I don't think my plan to divide the easy progression and conquer them is going to work. I can see regularity, but I can't find the patterns."

Keep trying different reorganizations and function applications. Some patterns require discovering an intermediate concept first.

### Special Cases and "Knock-outs"

Some leaves or piles have special behavior that doesn't fit the general formula:

- Document these explicitly
- Look for whether they can be expressed as boundary conditions
- The "knock-out" leaves in domain functions are an example

### Hardcoded Magic Numbers

When you see numbers like `6`, `4`, `2` in formulas, try to express these in terms of semantic functions if possible, but
sometimes a literal is the correct answer.

## Reproducing This Process

When asked to find patterns in static data:

1. **Request or locate the static data** in a clear format
2. **Identify the target**: what relationship are we trying to generalize?
3. **Apply the function catalog** systematically
4. **Try multiple reorganizations** of the data
5. **Look for binary/bit-level structure** in all values
6. **Write verbose comments** capturing every observed relationship
7. **Iterate**: delete unhelpful comments, expand helpful ones
8. **Build formula incrementally**: base case first, then corrections
9. **Test exhaustively** against the original data
10. **Generalize** by replacing literals with dimension-parameterized expressions

## Appendix: Key Files for Pattern Discovery

- `mapFolding/_e/_measure.py` — Bit-level measurement functions
- `mapFolding/_e/_semiotics.py` — Semantic constants and dimension-indexed functions
- `mapFolding/_e/_dataDynamic.py` — Domain and range functions (examples of generalized patterns)
- `mapFolding/_e/__init__.py` — Re-exports (the public API for these tools)
- `mapFolding/dataBaskets.py` — State classes with `productsOfDimensions`, `leavesTotal`, etc.

## Appendix: The Transformation Summary

| Before (Static)                    | After (Dynamic)                                                    |
| ---------------------------------- | ------------------------------------------------------------------ |
| 60+ explicit tuples                | Single nested loop                                                 |
| Valid only for 6D                  | Valid for any `dimensionsTotal >= 4`                               |
| Hardcoded `pileFirst` values       | `magicalSequence[indexUniversal] + 6 - (pileStepAbsolute * (...))` |
| Hardcoded `leafPredecessor` values | `Z0Z_invert(state, magicalSequence[...])` and arithmetic           |
| No visible structure               | Clear series structure via loop bounds                             |

The generalized code is simultaneously:

- **More compact** (fewer lines)
- **Self-documenting** (loop structure mirrors data structure)
- **Extensible** (works for 7D, 8D, etc. without modification)
