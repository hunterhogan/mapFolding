# Brain dump of things I wish were different

## Decorators, signatures, and docstrings

I wish @cache and @curry didn't obscure the function signature and docstring.

## "./easyRun" functions

These functions have matured, and I'll probably be using them for the foreseeable future. But their style diverges from the rest
of the codebase, and I wish they were more consistent with it. They are not dry: I copy-paste the code from one module to another.

## `toolz.dicttoolz.keyfilter` and `valfilter` don't do type narrowing

In the following function, for example, `valfilter` removes all `xmpz` types from the dictionary, but Pylance thinks the return type is wrong.

```python

def oopsAllLeaves(leavesPinned: int | xmpz) -> dict[int, int]:
    return valFilter(thisIsALeaf, leavesPinned)
```

> Type "dict[int, int | xmpz]" is not assignable to return type "dict[int, int]"

## Share transcription of Lunnon 1971

In Z0Z_literature\Lunnon1971.txt, I have transcribed most of the image-only PDF into text.

## Development tools

### VS Code importing deprecated types from typing instead of collections

Fix that shit already. I'm waiting for a reboot to confirm, but I might have a workaround. When I first install a venv, Pylance
uses the types from pandas in "site-packages", but if I open my pandas stub file that correlates with the pandas class/function in
my code, then Pylance seems to magically prefer all of my stub files in my directory over "site-packages". And it seems to stay
that way until I create a new venv. This is despite already setting a value for custom types in Pylance. So I am going to try the
trick with "C:\apps\mapFolding\typings\stdlib\_collections_abc.pyi".

### A Python formatter that formats my style

There seem to be some tools for creating formats that aren't "Black" or "PEP 8", but they all seem to be a huge pain in the ass.

### Simple sorting of functions

I can easily sort lines. But I want a fast way to alpha sort functions.

### Font

I very much like Fira Code and there might be other fonts I would like. I haven't looked because Fira Code is great. But Fira Code
doesn't seem to have the CJK code points I need, so I currently have two monospaced fonts with different widths. I just want the
same width for everything. Maybe I could change the pitch/CPI of the CJK font to match Fira Code?

### An easier way to type CJK

I use code snippets to enter ideograms, which is limiting and annoying. I haven't been able to get alt-codes to work.

### `__init__.py` and `# isort: split`

I wish I could use a global setting that says "don't sort the modules, but sort the symbols within each module".

### Ruff

In general, I love it. But I need to have different "levels" of strictness for different files. When I am developing a file or
function, I don't want diagnostics about commented out code or print or a bunch of other things.

### Reduce the settings files in the repo root

.editorconfig

#### mapFolding.code-workspace

move to "./.vscode"
mapFolding.code-workspace

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
