# Brain dump of things I wish were different

## Problems with Python installations, Windows, and VS Code

### At cmd, "python" forces a new terminal and it doesn't use the venv

This is a relatively new problem.

### In VS Code, Python environments picks 3.14z

It always chooses 3.14z as the default, and it overwrites the setting if I change it.

## Consolidate and organize knowledge in "Elimination.md"

- mapFolding\_e\analysisPython\Z0Z_p2d6.py
- mapFolding\_e\analysisPython\Z0Z_hypothesis.py
- mapFolding\_e\knowledgeDump.py
- NOTE statements
- Docstrings

## "./easyRun" functions

These functions have matured, and I'll probably be using them for the foreseeable future. But their style diverges from the rest
of the codebase, and I wish they were more consistent with it. They are not dry: I copy-paste the code from one module to another.

## Have clarity about allocating responsibility for the ordering of data

Who is "responsible" for putting data in order? And, when ought the order be predictable?

`getDomain二零and二` is returning an iterable of 2-tuple, and the 2-tuple is in a specific order: `(pileOfLeaf二零, pileOfLeaf二)`. If the function didn't want to order the 2-tuple, it would need to the data with a dictionary or a named tuple. It's cheaper and easier to use a 2-tuple and use the function identifier to signal the order: `getDomain二零and二` the first element is `pileOfLeaf二零`, and the second element is `pileOfLeaf二`.

Let's assume `getDomain二零and二` also tried to put the 2-tuples in a specific order, what would that order be? Rich-comparison ascending? Why not descending? It may be "natural" to sort a 2-tuple by the elements [0] then [1], but the elements in this case are leaf5 and leaf4: is it "natural" to sort the 2-tuples by leaf5 then leaf4?

That is especially important with `getDomain二一零and二一`. The 2-tuple is (leaf7, leaf6). But the only function that really uses that
domain is `getDomainDimension二`, which creates and returns a 4-tuple with order (leaf二一, leaf二一零, leaf二零, leaf二): leaf6
precedes leaf7 in the 4-tuple. Which sorting by `getDomain二一零and二一` is more "natural"? Sorting the group of 2-tuples based on

1. the indices of the elements in the 2-tuples,
2. the semantic value of the elements, specifically the values of leaf6 then leaf7 because of the ordinality of the elements, or
3. the expected use of the group of 2-tuples?

I don't think any of those answers are satisfying, so I think this is proof by contradiction: because there isn't a "natural" or a required order, I think the function should disavow a predictable order.

Should I force myself to consider the order of returned data and document my decision in the "Returns" section of the docstring? Yes, and I will permit myself to write "I don't know."

## `PileRangeOfLeaves` identifier

I'm not satisfied with `PileRangeOfLeaves`, but the "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo" problem limits my options.

- `range` is obfuscated due to `range()`.
- `leaves` is ambiguous.

Right now, I have `PermutationSpace = dict[int, LeafOrPileRangeOfLeaves]`. A good replacement for `PileRangeOfLeaves` would have the side effect of allowing me to create three _useful_ type aliases for three related dictionaries:

- ☑️ `PinnedLeaves = dict[pile, leaf]`
- `dict[pile, pileRangeOfLeaves]`
- ☑️ `PermutationSpace = dict[pile, leafOrPileRangeOfLeaves]`
