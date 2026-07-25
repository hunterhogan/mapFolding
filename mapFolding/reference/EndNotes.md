# Notes about the code

## absurd

`getLeavesTotal`: this check is one-degree short of absurd, but three lines of early absurdity is better than invalid output later. I'd add more checks if I could think of more. Fail early.

## pinning

The ONLY valid way to pin a `Leaf` in a `PermutationSpace` or `Folding` is to call a method of `PermutationSpace`.

## sorted

`PermutationSpace.addMissingLeafOptions()`: `sorted` overrides the insertion order and sorts based on `Pile` index. This is partially "defensive" in the sense that it is a consistent, logical, expected order, and may prevent odd results if another subroutine didn't guarantee the order when it ought to have. I'm hoping it improves efficiency, too.

## sortingDimensions

I previously sorted the dimensions for a few reasons that may or may not be valid:

1. After empirical testing, I believe that (2,10), for example, computes significantly faster than (10,2).
2. Standardization, generally.
3. If I recall correctly, after empirical testing, I concluded that sorted dimensions always leads to
non-negative values in the connection graph, but if the dimensions are not in ascending order of magnitude,
the connection graph might have negative values, which as far as I know, is not an inherent problem, but the
negative values propagate into other data structures, which requires the datatypes to hold negative values,
which means I cannot optimize the bit-widths of the datatypes as easily. (And optimized bit-widths helps with
performance.)

Furthermore, now that the package includes OEIS A000136, 1 x N stamps/maps, sorting could distort results.

## TypeAlias

Use `type` by default. Switch to `TypeAlias` whenever it promotes self-documenting code, especially through semiotics. Examples, I prefer `isinstance(x, LeafOptions)` to `isinstance(x, mpz)`; `dimension = DimensionIndex(2)` is more self-documenting than `dimension = int(2)`.

## walrus

Using the walrus operator here `if not (permutationSpace := _reduceLeafSpace...` means that type checkers are ok with `permutationSpace: PermutationSpace`. If I assigned without the `if` check, `permutationSpace = _reduceLeafSpace...`, then the annotation would need to be `permutationSpace: PermutationSpace | None` because `_reduceLeafSpace` can return `None`. Furthermore, not creating an intermediate variable is more efficient.
