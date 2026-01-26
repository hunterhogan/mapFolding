# Code style guidelines

## Always `<`

I've been standardizing on `<` or `<=`, not `>`.

## Replace "trailing operators" and "hidden" operators

This concept is a form of "A very broad concept: foreshadow from left to right and from top to bottom."

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
