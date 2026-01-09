# Standardizing instructions, identifiers, descriptions, and other semiotic elements for clarity

## Replace "trailing operators", "hidden" operators, and semantically-useless operators

### Trailing operators

#### Definition

There is certainly a better term than "trailing operators". Ideas:

- non-sequential operators
- dangling operator: as an analogy to dangling modifiers in English grammar
- surprise operators
- un-anticipated operators

#### Explanation

Computers parse and understand statements starting from the innermost nesting. Humans cannot naturally do that: we more easily
parse and understand a statement LTR or RTL. In `state.productsOfDimensions[dimensionNearest首(leafAt一零) - 1]`, we see these elements from LTR:

1. state
2. .productsOfDimensions
3. []
4. dimensionNearest首
5. ()
6. leafAt一零
7. - 1

At step 3, we learn we will modify state.productsOfDimensions.
At step 4, we know we will use the function's return.
But at step 7, we learn we need to modify the function's return.

`getitem(state.productsOfDimensions, (dimensionNearest首(leafAt一零) - 1))`

At step 1, we already know the first argument will be modified by the second element, so we are looking for those two elements.
At step 5, we know that whatever element is next will be combined with at least one more element, so we are not surprised by `-1`.

1. getitem
2. ()
3. state
4. .productsOfDimensions,
5. ()
6. dimensionNearest首
7. ()
8. leafAt一零
9. - 1

In simple statements, we don't need signals such as `getitem`, and excessive parentheses can make the statement more confusing.

In complex statements, use signals that allow the human to read LTR and anticipate later elements. Useful signals for complex statements include:

- parentheses
- `getitem`
- `DOTvalues` (for `dict.values()`)

### "hidden" operators

In complex statements, some operators are easy to miss, such as `~` and `-` when used to mean "negative". `operator.invert` and
`operator.neg` can ensure the human sees the operator.

### Semantically-useless operators

In the expression `range(bottles + 1)`, why is there a `+ 1`? We can often figure it out by analyzing the context, but the easiest way to erase ambiguity is to replace `+ 1` with a semantic identifier. In my packages look for modules named "_semiotics.py" to find replacements such as:

```python
decreasing: int = -1
"""Adjust the value due to Python syntax."""
inclusive: int = 1
"""Include the last value in a `range`: change from [p, q) to [p, q]."""
zeroIndexed: int = 1
"""Adjust the value due to Python syntax."""
```

In the above example, `range(bottles + 1)`, `range(bottles + inclusive)`, and `range(bottles + zeroIndexed)` all have the same effect, but semantic identifiers explain why the adjustment is needed.

## Semiotics: 2^n-dimensional

Previously, I used terms such as 2^d, 2Dn, p2d6. Furthermore, some (all?) academic papers use similar terms. But "d" is not obvious to all readers as "dimension". Therefore, I am standardizing on "2^n-dimensional" and `2上nDimensional`.

- Text
  - Morpheme root: 2^n-dimensional
  - Alternatives:
    - 2^n-dimensions
- Identifiers, pseudo-identifiers (e.g., function parameters), and file system objects
  - Morpheme suffix: 2上nDimensional
  - Always use a string that would be a valid Python identifier even if the string is not being used as an identifier.
  - Alternatives:
    - 2上n
  - Fallback prefix: p
    - p2上nDimensional
    - p2上n

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

## Needs more analysis or more details

### pick: always `<` or always `>`

I've been standardizing on `<` and `<=`, but `==` sometimes looks odd. Would `>` be better? I doubt it.

### `if`-trees of comparisons

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

### Replace `k` and `r` in most places with something semantic

It's probably ok or maybe preferable in "iff.py", but other than that, the reader must know the math conventions to understand what `k` and `r` are.

### Semantics of oddness/evenness apathy

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
