# Standardizing instructions, identifiers, descriptions, and other semiotic elements for clarity

## pick: always less that or always greater than

I've been standardizing on `<` and `<=`, but `==` sometimes looks odd. Would `>` be better? I doubt it.

## `if` trees of comparisons

example:

```python
   if dimension == 0:
    ...
   if dimension < state.dimensionsTotal - 2:
    ...
   if 0 < dimension < state.dimensionsTotal - 2:
    ...
   if 0 < dimension < state.dimensionsTotal - 3:
    ...
   if 0 < dimension < state.dimensionsTotal - 1:
    ...
```

If the order matters, either use flow control to enforce the order or document what matters and why.

If the order doesn't matter, then pick a standard order and use it everywhere.

## Replace `k` and `r` in most places with something semantic

It's probably ok or maybe preferable in "iff.py", but other than that, the reader must know the math conventions to understand what `k` and `r` are.

## Replace "trailing operators", "hidden" operators, and semantically void operators

Example

```python
anInteger: int = intInnit([integerNonnegative], 'integerNonnegative', type[int])[0]
```

It is very easy to miss the purpose of the list brackets--or that the indexer is even there. As a comparison, I've replaced
`dict.values()` with `DOTvalues` in many places, and I like it. In `astToolkit`, I have struggled with an indexing problem that
almost certainly has some valuable lessons for this specific index issue. Ironically, or perhaps appropriately, I think the
relevant code is in class `DOT`; no, wait, I think it's in class `Grab`.

## Semiotics: 2^n-dimensional

- Text
  - 2^n-dimensional
  - 2^n-dimensions
- Identifiers, pseudo-identifiers (e.g., function parameters), and file system objects
  - 2上n
  - 2上nDimensional
  - p2上n
  - p2上nDimensional

2^d, 2Dn, p2d[x]: "d" is not obvious to all readers as "dimension". Define d as dimension wherever it appears.

- 2^d-dimensional
- mapShapeIs2上nDimensions
- 2^n-dimensional (semantic similarities with `ndarray`, "n-dimensional array" from NumPy)
- 2^n-dimensions

## Example: OEIS

`def testCaseOeisFormula(request: pytest.FixtureRequest) -> TestCase:`

- "OeisFormula" is a diminutive form of oeisIDbyFormula: NO MOTHERFUCKING DIMINUTIVES.
- "OeisFormula" is referencing a very specific item, the module `oeisIDbyFormula`, and it is not a generalized form that includes
  `oeisIDbyFormula`, which means `oeisIDbyFormula` is used as a proper noun in this case: use the proper noun in the identifier.
- "Oeis" is not a word: use 'oeis' or 'OEIS' but not OeIs, oEIs, oeiS, or Oeis.

## Example of stupid error message

```python
if testCase.oeisID not in dictionaryOEISMapFolding:
    message: str = f"`{testCase.oeisID}` does not define a map shape."
```

The basic thesis of the error message that was triggered by `if testCase.oeisID not in dictionaryOEISMapFolding:` ought to be
"`testCase.oeisID` is not in `dictionaryOEISMapFolding`, therefore ..."

## Identifiers: past to future, LTR; cause to effect, LTR; general to specific, LTR

`mapShapeFromTestCase`

So testCase to mapShape, not mapShape from testCase.

## Semantics of oddness/evenness apathy

Be definition, for all odd numbers `dimensionNearestTail == 0`. Therefore, in cases like the one below and is some other cases
when I want to perform a test/measurement at the tail and I want to ignore oddness/evenness, I do not have a good standardized
way of handling it. From a programming perspective, bit shift is a super simple way to handle this. I can do the following with
any* number, `nn = (nn >> 1) << 1`, and the LSB will be `0`, which is what I want in this case and in many other cases.

(*any: 1) non-negative numbers only, and 2) I _feel_ like the number `1` is a special case because it _feels_ like all
information is destroyed as opposed to just some information being destroyed; it's not the same as `0` because no information is
destroyed if the original number is 0 or even. But analytically, I cannot find a tangible difference between `1` and other
non-negative values.)

In the "elimination" algorithms, I have banned the use bit shift. The positional-number notation is a proxy for Cartesian
coordinates, and bit shift doesn't correspond to an operation in Cartesian space (that I know about). I put a lot of effort into
using operations and functions that use the benefits of positional-number notation but semantically describe, uh, geometry, I
guess. `dimensionNearestTail` is a good example of the semantics: the function, `dimensionNearestTail` measures a
well-established concept--count trailing zeros (CTZ). Nevertheless, I have renamed the function so that even if the reader is
familiar with CTZ, they will not look at this geometry problem through the lens of "trailing zeros".

Therefore, I am conflicted about the use of bit shift, for example. However, it just occurred to me that I might be able to
"hide" the bit shifts inside functions. I've already done that with `howManyDimensionsHaveOddParity`, for example: the function
is merely bit_count()-1, but the name describes the semantics of what the function measures. I can probably do the same thing in
situations when I want to ignore oddness/evenness.

```python
  if is_odd(leafAt二):
    if dimensionNearestTail(leafAt二 - 1) == 1:
```
