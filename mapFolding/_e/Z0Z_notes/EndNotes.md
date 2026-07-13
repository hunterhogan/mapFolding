# Notes about the code

## absurd

`getLeavesTotal`: this check is one-degree short of absurd, but three lines of early absurdity is better than invalid output later. I'd add more checks if I could think of more. Fail early.

## pinning

The ONLY valid way to pin a `Leaf` in a `PermutationSpace` or `Folding` is to call a method of `PermutationSpace`.

## sorted

`PermutationSpace.addMissingLeafOptions()`: `sorted` overrides the insertion order and sorts based on `Pile` index. This is partially "defensive" in the sense that it is a consistent, logical, expected order, and may prevent odd results if another subroutine didn't guarantee the order when it ought to have. I'm hoping it improves efficiency, too.

## TypeAlias

Use `type` by default. Switch to `TypeAlias` whenever it promotes self-documenting code, especially through semiotics. Examples, I prefer `isinstance(x, LeafOptions)` to `isinstance(x, mpz)`; `dimension = DimensionIndex(2)` is more self-documenting than `dimension = int(2)`.

## walrus

Using the walrus operator here `if not (permutationSpace := _reduceLeafSpace...` means that type checkers are ok with `permutationSpace: PermutationSpace`. If I assigned without the `if` check, `permutationSpace = _reduceLeafSpace...`, then the annotation would need to be `permutationSpace: PermutationSpace | None` because `_reduceLeafSpace` can return `None`. Furthermore, not creating an intermediate variable is more efficient.
