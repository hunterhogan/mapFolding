# mapFolding

`mapFolding` is a typed Python research package for exact enumeration of map foldings, stamp foldings, semi-meanders, and meanders. It provides counting algorithms, access to their computation states, and tools for transforming readable algorithm source into specialized implementations.

[![Install](https://img.shields.io/badge/Install-mapFolding-2ea44f?style=for-the-badge)](#installation)
[![PyPI](https://img.shields.io/pypi/v/mapFolding.svg)](https://pypi.org/project/mapFolding/)
[![Python versions](https://img.shields.io/pypi/pyversions/mapFolding.svg)](https://pypi.org/project/mapFolding/)
[![Python Tests](https://github.com/hunterhogan/mapFolding/actions/workflows/pythonTests.yml/badge.svg)](https://github.com/hunterhogan/mapFolding/actions/workflows/pythonTests.yml)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)](https://github.com/hunterhogan/mapFolding/blob/main/LICENSE)

A map-folding problem asks how many ways the labeled panels of a map can be stacked while preserving their connections. A strip of stamps is the one-dimensional case. Related meander problems count configurations of non-self-intersecting curves crossing a line or ray.

## Research results

I use `mapFolding` to compute new sequence terms and develop exact relationships between folding and meander problems.

- 2 × n maps: I computed the folding counts for `2 × 19` and `2 × 20` stamp strips, extending [OEIS A001415](https://oeis.org/A001415). OEIS published the terms in 2025 March.
- Semi-meanders: I extended the semi-meander sequence [OEIS A000682](https://oeis.org/A000682) from `n = 45` through `n = 49` and added many formulas relating it to stamp foldings, meanders, and other sequences. The package's [`byFormula.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/oeis/byFormula.py) implements exact sequence relationships for further computation and comparison.

## Research with mapFolding

You can use the package to calculate sequence terms, dissect an algorithm's state, develop a different counting method, or study the effects of source transformations. The main routes through the code are:

| Work                                 | Where to begin                                                                                                                                                                                                                                                 |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Count foldings or meanders           | [`basecamp.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/basecamp.py) selects an algorithm and prepares its state.                                                                                                                       |
| Work with integer sequences          | [`oeis/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/oeis/) provides sequence dispatch, exact formula relationships, metadata, known values, and triangle utilities.                                                                       |
| Inspect or change a computation      | [`dataBaskets.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/dataBaskets.py) describes the state containers; [`algorithms/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/algorithms/) contains source implementations. |
| Transform or specialize an algorithm | [`kitAST/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/kitAST/) generates variants and jobs, with derived implementations in [`synthesized/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/synthesized/).                |
| Explore a different search strategy  | [`algorithms/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/algorithms/) contains elimination, constraint propagation, and insertion; package modules provide search-space analysis.                                                        |
| Validate a new implementation        | [`test_algorithms.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/tests/test_algorithms.py) compares implementations with known sequence values.                                                                                           |

The module, function, and class docstrings explain parameters, state layouts, mathematical relationships, and implementation details. Use them when choosing an experiment or following a computation through the package.

## Counting and sequence access

The main counting functions are in `mapFolding.basecamp`; sequence functions are in `mapFolding.oeis`.

| Task                                                   | Interface                            | Result                                              |
| ------------------------------------------------------ | ------------------------------------ | --------------------------------------------------- |
| Count foldings for supplied dimensions                 | `countFolds(mapShape, ...)`          | Integer count                                       |
| Count foldings by elimination or constraint solving    | `eliminateFolds(mapShape, ...)`      | Integer count                                       |
| Count rotationally symmetric stamp foldings            | `countFoldsSymmetric(mapShape, ...)` | Integer count                                       |
| Count semi-meanders, open meanders, or closed meanders | `countMeanders(kind, n, ...)`        | Completed `StateMeanders`                           |
| Calculate a supported OEIS term                        | `oeisIDfor_n(oeisID, n, ...)`        | Integer value of `a(n)`                             |
| Convert a folding-sequence index to dimensions         | `makeMapShape(oeisID, n)`            | Dimension tuple                                     |
| Retrieve known sequence values                         | `getValuesKnown(oeisID)`             | Dictionary of indices and values                    |
| Inspect sequence metadata                              | `getMetadata(oeisID)`                | Description, offset, known values, and row settings |

For example, `(2, 6)` describes a `2 × 6` map, and `(2, 2, 2)` describes a three-dimensional map with eight panels. Dimensions count panels along each axis.

### Map dimensions and OEIS indices

| Sequence                            | Problem                                                  | `mapShape` at index `n` |
| ----------------------------------- | -------------------------------------------------------- | ----------------------- |
| [A000136](https://oeis.org/A000136) | Strip of `n` labeled stamps                              | `(1, n)`                |
| [A001415](https://oeis.org/A001415) | `2 × n` strip                                            | `(2, n)`                |
| [A001416](https://oeis.org/A001416) | `3 × n` strip                                            | `(3, n)`                |
| [A001417](https://oeis.org/A001417) | `n`-dimensional map with every dimension of length two   | `(2,) * n`              |
| [A195646](https://oeis.org/A195646) | `n`-dimensional map with every dimension of length three | `(3,) * n`              |
| [A001418](https://oeis.org/A001418) | `n × n` sheet                                            | `(n, n)`                |
| [A007822](https://oeis.org/A007822) | Symmetric foldings of `2n + 1` stamps                    | `(1, 2 * n)`            |

For A007822, `countFoldsSymmetric` uses an even-length computational strip to count the symmetric foldings of the odd-length strip in the sequence description. `makeMapShape` supplies that computational shape.

### Meanders and computation state

`countMeanders('semi', n)` computes [A000682](https://oeis.org/A000682), and `countMeanders('meanders', n)` computes the open-meander sequence [A005316](https://oeis.org/A005316). The `'closed'` selector computes [A005315](https://oeis.org/A005315) using the corresponding odd-index open-meander term.

The returned state retains the boundary position and transfer counts for inspection. Read the
completed count from its `totalMeanders` property:

```python
state.totalMeanders
```

`totalMeanders` sums the active lookup counts and `countAddend`, which retains completed
contributions removed by pruning. The property reflects the current state; before counting finishes,
its value represents intermediate counts rather than the final sequence term.

You can also supply an existing `StateMeanders` through the `state` argument to continue that computation. See the state and counting-function docstrings for the relevant fields and behavior.

### Sequence formulas and reference data

`oeisIDfor_n` dispatches to a folding, symmetric-folding, or meander algorithm, or to an exact formula connecting related sequences. For sequences with selectable identities, its `f` argument chooses a formula. The formulas and their docstrings are in [`oeis/byFormula.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/oeis/byFormula.py).

The `getOEISids` console command lists registered sequences and usage examples. The registry also includes related sequences used as reference data; consult `oeisIDfor_n` and `byFormula` for computation support. The `OEIS_for_n` command evaluates a supported term from its sequence identifier and index.

OEIS metadata and b-files are loaded when the metadata module is imported. Local files have a 30-day freshness interval; missing or stale entries trigger a request to `oeis.org`, and stale data remains available if a refresh fails.

### Triangle data and new terms

Triangle utilities use the sequence metadata to locate rows, columns, and flat b-file indices. Columns start at one, including when a row is supplied as a Python sequence.

For example, [A400429](https://oeis.org/A400429) organizes semi-meanders by crossings and winding number; each row sums to the corresponding A000682 term. [`oeis/A400429/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/oeis/A400429/) exposes triangle-cell identities and fixed-diagonal formulas. Its documentation distinguishes the proved first four diagonals from the conjectured formulas for diagonals five through eleven.

| Task                         | Interface                                                 | Result                                                    |
| ---------------------------- | --------------------------------------------------------- | --------------------------------------------------------- |
| Read consecutive known rows  | `getTriangleRows(oeisID)`                                 | Dense rows, stopping at the first missing flat index      |
| Preserve gaps in known terms | `getTriangleTerms(oeisID)`                                | Rows mapped to dictionaries of available columns          |
| Select a diagonal            | `getTriangleDiagonal(oeisID, 次diagonal, fromRight=True)` | Available terms keyed by row number; offsets start at one |
| Locate a row in a b-file     | `getTriangleIndices(oeisID, n)`                           | Range of flat indices for the complete row                |
| Save newly computed terms    | `updateBFile(oeisID, sequence)`                           | Local b-file path                                         |
| Save dense or sparse rows    | `updateTriangle(oeisID, triangle)`                        | Local b-file path                                         |

`updateBFile` and `updateTriangle` add missing terms to the local cache and update the shared known-value dictionary in place. Existing local b-file values take precedence over metadata values, which take precedence over supplied values. These functions retain b-file comments and write terms in index order; they do not submit edits to OEIS. `valueUnknown` is one greater than the largest stored index, so it can lie beyond gaps in a sparse sequence.

For research files, `mapFolding.kitFilesystem` reads and writes triangle CSV records with a row key followed by cell values. `writeTriangle` leaves empty fields for missing sparse cells; use `readTriangleSparse` to preserve their column positions. `readTriangle` expects every cell field to contain an integer. Diagonal CSV files use three fields: row key, diagonal offset, and value.

`printEasyRunBenchmark` compares one computed term with its known value. `printEasyRunTriangle` compares a computed row with its known cells and aligns differing rows for inspection. Both accept `ratio` as a number of decimal places; `ratio=0` hides ratios.

## Algorithm experiments

The package includes several approaches to the same combinatorial problems:

- **Map folding:** insertion-search algorithms derived from Lunnon's work, including variants that apply symmetry restrictions from his theorems.
- **Meanders:** transfer-matrix counting based on Jensen's approach, with dictionary, NumPy, Pandas, and pruning variants.
- **Permutation generation:** stamp-folding and meander algorithms based on Sawada and Li's work, with experimental bilateral variants.
- **Elimination and insertion:** elimination, constraint propagation, crease-based search, and incremental arch-diagram generation in [`algorithms/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/algorithms/).

`mapFolding.basecamp.eliminateFolds` selects the `'elimination'`, `'constraintPropagation'`, or
`'crease'` flow. It accepts a map shape or an existing `mapFolding.dataBaskets.StateElimination`,
which groups permutation spaces, folding representatives, and symmetry multipliers. Each
`PermutationSpace` maps pile positions to pinned leaves or `ChoicesLeaf` values.

The supporting operations are package modules: [`leafDomains.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/leafDomains.py)
computes possible piles for leaves, [`pileOptions.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/pileOptions.py)
computes possible leaves for piles, and [`beDRYElimination.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/beDRYElimination.py)
constructs and inspects `ChoicesLeaf` values. [`pinIt.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/pinIt.py)
restricts assignments and relative leaf order; [`reduceIt.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/reduceIt.py)
propagates constraints through the search spaces. Specialized rules for maps whose dimension lengths
are all two use `conditionalOrdering`, `creases`, `measure`, `pinByCrease`, `pinByDomain`, and the
positional-numeral notation in `semiotics`.

Counting entry points accept a `flow` preference. Omit it to use the default route. Available variants and dispatch rules change as the research develops; [`basecamp.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/basecamp.py) and the relevant `easyRun` runner show the current choices.

The transformation pipeline in `kitAST` can inline functions, decompose state dataclasses, specialize initialized values for a map shape, and prepare Numba or Linux Codon jobs. [`RecipeJob.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/kitAST/RecipeJob.py) describes job configuration; [`easyRun/synthesizeModules.py`](https://github.com/hunterhogan/mapFolding/blob/main/easyRun/synthesizeModules.py) coordinates module regeneration. For changes that should survive regeneration, edit the source algorithm or its transformation.

General-purpose transformation tools developed alongside this project live in [astToolkit](https://github.com/hunterhogan/astToolkit) and [astToolFactory](https://github.com/hunterhogan/astToolFactory). The transformations retained here serve `mapFolding` algorithms and jobs.

### Computation cost and saved results

Exact enumeration becomes expensive quickly. Running time, memory use, and the appropriate implementation depend on the problem and its size; measure smaller cases before committing to a long computation.

Pass `pathLikeWrite` to a counting function to save the total. An existing directory receives a generated filename; an explicit filename is also supported. The destination is write-tested before counting, leaving a test value until the result replaces it. An existing target file is rejected. These output files save totals; they do not checkpoint a running search.

`countFolds` can divide work using an integer, `computationDivisions='cpu'`, or `computationDivisions='maximum'`. Tasks repeat substantial preparatory work, so division can increase elapsed time. Leave `computationDivisions=None` unless task division is part of your experiment; see the function's docstring for the tradeoffs.

## Repository guide

| Path                                                                                                                                                                                                                                                                                                           | Contents                                                                   |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| [`mapFolding/dataStructures.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/dataStructures.py), [`theTypes.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/theTypes.py), and [`theSSOT.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/theSSOT.py) | Array storage, domain types, and shared settings                           |
| [`mapFolding/kitFilesystem.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/kitFilesystem.py)                                                                                                                                                                                               | Result files, caches, and computation-data storage                         |
| [`mapFolding/jobs/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/jobs/)                                                                                                                                                                                                                     | Computation jobs specialized for particular map shapes                     |
| [`mapFolding/tests/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/tests/)                                                                                                                                                                                                                   | Correctness checks, dispatch and filesystem tests, and stored data samples |
| [`easyRun/`](https://github.com/hunterhogan/mapFolding/tree/main/easyRun/)                                                                                                                                                                                                                                     | Editable runners for computations, benchmarks, synthesis, and exploration  |
| [`research/elimination/`](https://github.com/hunterhogan/mapFolding/tree/main/research/elimination/) and [`research/`](https://github.com/hunterhogan/mapFolding/tree/main/research/)                                                                                                                          | Analysis tools, folding data, and research materials                       |
| [`archive/`](https://github.com/hunterhogan/mapFolding/tree/main/archive/) and [`development/`](https://github.com/hunterhogan/mapFolding/tree/main/development/)                                                                                                                                              | Historical code, notes, and development materials                          |
| [`citations/`](https://github.com/hunterhogan/mapFolding/tree/main/citations/)                                                                                                                                                                                                                                 | Software citation metadata and BibTeX references                           |

## Citation and references

To cite this software, use [`CITATION.cff`](https://github.com/hunterhogan/mapFolding/blob/main/CITATION.cff). The mathematical and implementation references below have BibTeX entries in [`citations/`](https://github.com/hunterhogan/mapFolding/tree/main/citations/).

### Folding and meander algorithms

- **Albert Sadé (1949), _Sur les Chevauchements des Permutations_.** [PDF](https://oeis.org/A000108/a000108_17.pdf) · [BibTeX](https://github.com/hunterhogan/mapFolding/blob/main/citations/Sade1949.bib).
- **John E. Koehler (1968), _Folding a strip of stamps_.** [DOI](https://doi.org/10.1016/S0021-9800(68)80048-1) · [BibTeX](https://github.com/hunterhogan/mapFolding/blob/main/citations/Koehler1968.bib).
- **W. F. Lunnon (1968), _A map-folding problem_.** [DOI](https://doi.org/10.1090/S0025-5718-1968-0221957-8) · [PDF](https://pubs.ams.org/journals/mcom/1968-22-101/S0025-5718-1968-0221957-8/S0025-5718-1968-0221957-8.pdf) · [BibTeX](https://github.com/hunterhogan/mapFolding/blob/main/citations/Lunnon1968.bib).
- **W. F. Lunnon (1971), _Multi-dimensional map-folding_.** The basis for the multidimensional counting algorithms. [DOI](https://doi.org/10.1093/comjnl/14.1.75) · [PDF](https://academic.oup.com/comjnl/article-pdf/14/1/75/1020149/140075.pdf) · [BibTeX](https://github.com/hunterhogan/mapFolding/blob/main/citations/Lunnon.bib).
- **Iwan Jensen (2000), _A transfer matrix approach to the enumeration of plane meanders_.** The transfer-matrix method underlying the matrix-meander algorithms. [DOI](https://doi.org/10.1088/0305-4470/33/34/301) · [Preprint](https://arxiv.org/abs/cond-mat/0008178) · [TeX source](https://arxiv.org/src/cond-mat/0008178) · [BibTeX](https://github.com/hunterhogan/mapFolding/blob/main/citations/Jensen.bib).
- **Joe Sawada and Roy Li (2012), _Stamp Foldings, Semi-Meanders, and Open Meanders: Fast Generation Algorithms_.** Permutation-based generation methods used in the experimental permutation algorithms. [DOI](https://doi.org/10.37236/2404) · [PDF](https://www.combinatorics.org/ojs/index.php/eljc/article/view/v19i2p43/pdf) · [BibTeX](https://github.com/hunterhogan/mapFolding/blob/main/citations/Sawada2012.bib).
- **Stéphane Legendre (2014), _Foldings and meanders_.** Relationships between folding and meander sequences, with folding-validity conditions. [PDF](https://ajc.maths.uq.edu.au/pdf/58/ajc_v58_p275.pdf) · [Preprint](https://arxiv.org/abs/1302.2025) · [TeX source](https://arxiv.org/src/1302.2025) · [BibTeX](https://github.com/hunterhogan/mapFolding/blob/main/citations/Legendre2014.bib).

### Related research

- **Nathan Clisby and Iwan Jensen (2012), _A new transfer-matrix algorithm for exact enumerations: self-avoiding polygons on the square lattice_.** Further transfer-matrix techniques for exact counting, demonstrated on lattice polygons. [DOI](https://doi.org/10.1088/1751-8113/45/11/115202) · [Preprint](https://arxiv.org/abs/1111.5877) · [BibTeX](https://github.com/hunterhogan/mapFolding/blob/main/citations/ClisbyJensen2012.bib).
- **Yiyang Jia, Jun Mitani, and Ryuhei Uehara (2020), _Valid Orderings of Layers When Simple-Folding a Map_.** A linear-time test for whether a supplied layer ordering is valid under simple folding, providing context for the role of the folding model in validity tests. [DOI](https://doi.org/10.2197/ipsjjip.28.816) · [Article and PDF](https://www.jstage.jst.go.jp/article/ipsjjip/28/0/28_816/_article) · [BibTeX](https://github.com/hunterhogan/mapFolding/blob/main/citations/Jia2020.bib).

### Sequence data and other implementations

- **The On-Line Encyclopedia of Integer Sequences.** Sequence definitions, known values, formulas, and computation records. [OEIS](https://oeis.org/) · [BibTeX](https://github.com/hunterhogan/mapFolding/blob/main/citations/oeis.bib).
- **jOEIS: Java Online Encyclopedia of Integer Sequences.** [Repository](https://github.com/archmageirvine/joeis) · [Sean A. Irvine's A001415 implementation](https://github.com/archmageirvine/joeis/blob/9e2bd5f6b5a51bc40833c19ca04555ec12044edc/src/irvine/oeis/a001/A001415.java) · [BibTeX](https://github.com/hunterhogan/mapFolding/blob/main/citations/jOEIS.bib).
- **Andrew Howroyd's meander implementation.** [C# source and description](https://oeis.org/A005316/a005316.cs.txt) · [BibTeX](https://github.com/hunterhogan/mapFolding/blob/main/citations/Howroyd.bib).

## Installation

`mapFolding` requires Python 3.13 or newer. Install the published package from [PyPI](https://pypi.org/project/mapFolding/):

```console
pip install mapFolding
```

NumPy and Numba are included in the base dependencies. Optional extras add other implementations and development tools:

| Extra         | Purpose                                                                           |
| ------------- | --------------------------------------------------------------------------------- |
| `pandas`      | Pandas and Arrow meander implementations                                          |
| `codon`       | [Codon](https://docs.exaloop.io/)-compiled jobs on Linux                          |
| `ortools`     | Experimental constraint propagation; installed only on Python versions below 3.15 |
| `testing`     | Test dependencies and optional implementations exercised by the tests             |
| `development` | Profiling, notebooks, type stubs, and development tools                           |

For example:

```console
pip install "mapFolding[pandas]"
```

### Development and supplemental materials

Clone the repository to work on the algorithms or access its archives, notes, and research materials:

```console
git clone https://github.com/hunterhogan/mapFolding.git
cd mapFolding
python -m venv .venv
```

Activate the environment before installing. On PowerShell:

```powershell
. .\.venv\Scripts\Activate.ps1
```

On a POSIX shell:

```sh
. .venv/bin/activate
```

Then install the development and testing extras:

```console
pip install -e ".[development,testing]"
```

Run project commands from the repository root with the environment active. For example, to check the main counting implementations:

```console
pytest mapFolding/tests/test_algorithms.py
```

The tests compare algorithm variants with known OEIS values and exercise generated jobs. When adding a counting implementation, extend the relevant test cases in [`test_algorithms.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/tests/test_algorithms.py). Triangle conversions, OEIS updates, and sequence formulas have dedicated checks in `test_dataStructures.py`, `test_oeis.py`, and `test_oeis_byFormula.py`; supporting behavior has its own tests in `mapFolding/tests`.

## Computation targets

These are my computation plans and estimates for my hardware, including work with unpublished algorithms.

- Folds of multidimensional map (2, 2, 2, 2, 2, 2, 2, 2), which is [A001417(8)](https://oeis.org/A001417): estimated 14 days with my current hardware and a non-published algorithm.
- Folds of multidimensional map (2, 21), which is [A001415(21)](https://oeis.org/A001415): estimated 6 days with my current hardware and one of the compiled "Theorem 2" algorithm versions.
- Folds of multidimensional map (2, 22), which is [A001415(22)](https://oeis.org/A001415): estimated 15 days with my current hardware and one of the compiled "Theorem 2" algorithm versions.
- Folds of multidimensional map (3, 15), which is [A001416(15)](https://oeis.org/A001416): estimated 11 days with my current hardware and one of the compiled "Theorem 2" algorithm versions.
- Semi-meanders with 50 crossings, which is [A000682(50)](https://oeis.org/A000682): estimated 1–2 hours if I had 64 GB of system memory.

Computation times assume:

1. No power interruptions. (I don't have a UPS, and in the last two months the longest gap between power interruptions was 8 days 6 hours.)
2. No system crashes.
3. Single-core computations are given priority over other tasks.
4. The core of single-core computations is regularly boosted.
5. No heat issues.

## Maintainer's working notes

This is my ongoing to-do list across this package and related projects. It records experiments, unresolved problems, and reminders to myself.

1. Acceleration
    1. Codon: new release, new features.
2. Remote execution
    1. Google Colab: memmap causes everything to go to "disk" and the physical memory is unused.
    2. GitHub Codespaces: ignores memmap and terminates on OOM.
3. astToolkit: figure out a better container system than Ingredients Module, Ingredients Function, etc.
4. Dao of map folding
    1. Overhaul how I store and access type metadata for ndarray and other containers. Affects:
        1. StateMapFolding
        2. Shatter dataclass
        3. ast transformations
5. Symmetric 1xn
    1. Symmetry algorithm is too slow.
    2. Asynchronous + accelerated doesn't work yet.
6. Matrix Meanders
    1. Rebuild the Polars meander implementation because it's broken.
    2. I need a lot more tools in kitAST.
    3. I hate `getTotalBuckets` but it _should_ be a brilliant function.
7. Permutations
    1. The base algorithm is a dumpster fire.
    2. Bilateral concurrent can't produce a folding, but uses too much memory and is too slow to count folds.
8. Reduce arches: slower than counting by hand.
9. Constraint propagation or SAT
    1. ortools won't apply concurrency, and I don't know how to think about divisions of the problem space that actually reduce computation time.
    2. Understanding other packages is harder than a self-enema with a firehouse.
10. 2^n-dimensional (aka elimination by crease)
    1. Computation algorithms needed for: conditional predecessors, conditional successors,
     DomainLeaf首零Plus零 in dimension零, the range of a pile (aka pile options, aka choices
     leaf), pinPile二ByCrease, pinPile二Ante首ByCrease, and pinPile零Ante首零AfterDepth4.
    2. Acceleration.
    3. Bug in `boxOfFunctionsReduction2上nDimensional`.
    4. Finish collecting 2^7 data.
    5. Fastest algorithm for 2^5. Can't complete 2^6. I mean, wtf?
    6. `PermutationSpace` and its OOP structure: I don't really know what I am doing.
    7. The reduce-it concept is strong: reorient more code around it. Resolve the tension with `PermutationSpace._solidifyLeafSpace()`.
11. Elimination: atrophied.
12. Insert leaves: But for the data storage, it would be powerful. IDK anything about data compression. I lost the code I wrote for the notation I created to store graphs, and it seems I am too stupid to recreate or reverse engineer my own code from my own files AND OTHER FUNCTIONS. fml. Ironically, LLMs can't figure it out either because I guess my idea was novel. F.M.L.

Hunter, this list is incomplete: you quit writing after you got upset. Resume at mapFolding.kitAST.

## My recovery

[![2011 August: Homeless since](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)

[![CC-BY-NC-4.0](https://raw.githubusercontent.com/hunterhogan/mapFolding/refs/heads/main/.github/CC-BY-NC-4.0.png)](https://creativecommons.org/licenses/by-nc/4.0/)
