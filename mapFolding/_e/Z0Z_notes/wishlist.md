# Brain dump of things I wish were different

## More tests for computations

## Consolidate and organize knowledge in "Elimination.md"

- mapFolding\_e\analysisPython\Z0Z_p2d6.py
- mapFolding\_e\analysisPython\Z0Z_hypothesis.py
- mapFolding\_e\knowledgeDump.py
- NOTE statements
- Docstrings

## "./easyRun" functions

These functions have matured, and I'll probably be using them for the foreseeable future. But their style diverges from the rest
of the codebase, and I wish they were more consistent with it. They are not dry: I copy-paste the code from one module to another.

## `LeafOptions` identifier

I'm not satisfied with `LeafOptions`, but the "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo" problem limits my options.

- `range` is obfuscated due to `range()`.
- `leaves` is ambiguous.

Right now, I have `PermutationSpace = dict[int, LeafSpace]`. A good replacement for `LeafOptions` would have the side effect of allowing me to create three _useful_ type aliases for three related dictionaries:

- ☑️ `PinnedLeaves = dict[pile, leaf]`
- `dict[pile, leafOptions]`
- ☑️ `PermutationSpace = dict[pile, leafSpace]`
