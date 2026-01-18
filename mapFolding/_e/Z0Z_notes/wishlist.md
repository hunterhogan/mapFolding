# Brain dump of things I wish were different

## Real-time `listPermutationSpace`

As the pinning functions are working, I'd like to see the rate of change of `listPermutationSpace`.

## At cmd, "python" forces a new terminal and it doesn't use the venv

This is a relatively new problem.

## Consolidate and organize knowledge in "Elimination.md"

- mapFolding\_e\analysisPython\Z0Z_p2d6.py
- mapFolding\_e\analysisPython\Z0Z_hypothesis.py
- mapFolding\_e\knowledgeDump.py

## "./easyRun" functions

These functions have matured, and I'll probably be using them for the foreseeable future. But their style diverges from the rest
of the codebase, and I wish they were more consistent with it. They are not dry: I copy-paste the code from one module to another.

## Share transcription of Lunnon 1971

In Z0Z_literature\Lunnon1971.txt, I have transcribed most of the image-only PDF into text.

## Ideas

### Bifurcate `PermutationSpace` if a `PileRangeOfLeaves` has exactly two leaves

This is not a subtle implementation, but it might be useful. After `updateListPermutationSpacePileRangesOfLeaves`, something like
`(any(valfilter(bit_count == 3)), oopsAllPileRangesOfLeaves, state.listPermutationSpace)` to find all `PileRangeOfLeaves` with
exactly two leaves, then split the corresponding `PermutationSpace` into two `PermutationSpace` objects, replacing
`PileRangeOfLeaves` with `int`. Should I then run the new `PermutationSpace` back through
`updateListPermutationSpacePileRangesOfLeaves`? I _feel_ like `notEnoughOpenPiles`, for example, will eliminate some of the new
`PermutationSpace` objects, which is the point.

### Sophisticated bifurcation/separation of `PermutationSpace`

Many relationships cannot be expressed with `PileRangeOfLeaves`. In a 2^6 map, most of the time, leaf9 and leaf13 can be in any
order, but if leaf13 is in pile3, pile5, or pile7, then leaf9 must precede leaf13. If leaf13 is pinned, `_conditionalPredecessors`
will change the `PileRangeOfLeaves` and `notEnoughOpenPiles` might disqualify the `PermutationSpace`. Nevertheless, it _might_ be
advantageous to divide the `PermutationSpace` into four dictionaries:

1. pile3: leaf13
2. pile5: leaf13
3. pile7: leaf13
4. At pile3, pile5, or pile7, remove leaf13 from `PileRangeOfLeaves`.

Then other effects would cascade through the four dictionaries due to other functions.

### Make a 2^n-dimensional version of `thisLeafFoldingIsValid`

The math is far less complex with 2^n-dimensional maps: the computational savings might be multiple orders of magnitude.

## Development tools

### Pylance importing deprecated types from typing instead of collections

Fix that shit already. I'm waiting for a reboot to confirm, but I might have a workaround. When I first install a venv, Pylance
uses the types from pandas in "site-packages", but if I open my pandas stub file that correlates with the pandas class/function in
my code, then Pylance seems to magically prefer all of my stub files in my directory over "site-packages". And it seems to stay
that way until I create a new venv. This is despite already setting a value for custom types in Pylance. So I am going to try the
trick with "C:\apps\mapFolding\typings\stdlib\_collections_abc.pyi". This did not work.

### A Python formatter that formats my style

There seem to be some tools for creating formats that aren't "Black" or "PEP 8", but they all seem to be a huge pain in the ass.

### Simple sorting of functions

I can easily sort lines. But I want a fast way to alpha sort functions.

### Font

I very much like Fira Code and there might be other fonts I would like. I haven't looked because Fira Code is great. But Fira Code
doesn't seem to have the CJK code points I need, so I currently have two monospaced fonts with different widths. I just want the
same width for everything. Maybe I could change the pitch/CPI of the CJK font to match Fira Code? CJK glyphs are double the width.

### An easier way to type CJK

I use code snippets to enter ideograms, which is limiting and annoying. I haven't been able to get alt-codes to work.

### `__init__.py` and `# isort: split`

I wish I could use a global setting that says "don't sort the modules, but sort the symbols within each module".

### Ruff

In general, I love it. But I need to have different "levels" of strictness for different files. When I am developing a file or
function, I don't want diagnostics about commented out code or print or a bunch of other things.

### Reduce the settings files in the repo root

.editorconfig

#### isort and ruff

`hunterMakesPy` has a mechanism in Python for storing these settings so they can be used for `writePython()`.
.isort.cfg
ruff.toml

#### I think I'm stuck with these

.gitattributes
.gitignore
CITATION.cff
LICENSE
pyproject.toml
README.md
SECURITY.md
uv.lock

### grepWin

Integrate grepWin. Maybe nirSoft searchMyFiles, too.

### ss64.com knowledge

Batch file support sucks.
