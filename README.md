# mapFolding

[![PyPI](https://img.shields.io/pypi/v/mapFolding.svg)](https://pypi.org/project/mapFolding/)
[![Python versions](https://img.shields.io/pypi/pyversions/mapFolding.svg)](https://pypi.org/project/mapFolding/)
[![Python Tests](https://github.com/hunterhogan/mapFolding/actions/workflows/pythonTests.yml/badge.svg)](https://github.com/hunterhogan/mapFolding/actions/workflows/pythonTests.yml)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)](https://github.com/hunterhogan/mapFolding/blob/main/LICENSE)

Exact enumeration tools for map folding, stamp folding, semi-meanders, and meanders.

`mapFolding` is a typed Python research package for counting distinct foldings of one- and multidimensional maps. It provides:

- stable dispatch functions for ordinary, rotationally symmetric, and divided computations;
- native algorithms for map folding, semi-meanders, and meanders;
- a unified interface to 33 implemented [OEIS](https://oeis.org/) sequences, including exact formula relationships;
- readable source algorithms, generated optimized implementations, and optional NumPy, Numba, Pandas, and Codon routes; and
- cross-implementation tests against known sequence values.

Earlier versions of this project were used to compute new terms for [OEIS A001415](https://oeis.org/A001415), the number of ways to fold a `2 × n` strip of stamps.

This is exact combinatorial enumeration: running time and memory requirements grow quickly. Start with small inputs. The supported high-level interfaces are in `mapFolding.basecamp` and `mapFolding.oeis`; other modules include evolving research code.

## Installation

`mapFolding` requires Python 3.13 or newer.

```console
pip install mapFolding
```

Install an optional backend only when you need its corresponding implementation:

| Extra         | Purpose                                       |
| ------------- | --------------------------------------------- |
| `numba`       | Numba-compiled map-folding implementations    |
| `pandas`      | Pandas and Arrow meander implementation       |
| `codon`       | Codon-compiled implementations on Linux       |
| `ortools`     | Experimental constraint-propagation work      |
| `testing`     | Test dependencies                             |
| `development` | Broader development and analysis dependencies |

For example:

```console
pip install "mapFolding[numba,pandas]"
```

## Quick start

### Count map foldings

Each positive integer in `mapShape` is the length of one dimension. A `(2, 3)` map has six leaves.

```python
from mapFolding.basecamp import countFolds

folds_total = countFolds((2, 3))
print(folds_total)  # 60
```

### Calculate an OEIS term

```python
from mapFolding.oeis import oeisIDfor_n

folds_total = oeisIDfor_n('A001415', 6)
print(folds_total)  # 10512
```

The installed commands expose the same sequence registry:

```console
getOEISids
OEIS_for_n A001415 6
```

The second command prints:

```text
10512 distinct folding patterns.
Time elapsed: ... seconds
```

### Count semi-meanders and meanders

```python
from mapFolding.basecamp import countMeanders

print(countMeanders('semi', 5))      # 10; OEIS A000682
print(countMeanders('meanders', 4))  # 3; OEIS A005316
```

## Public interfaces

| Goal                                           | Interface                                                |
| ---------------------------------------------- | -------------------------------------------------------- |
| Count all foldings of a map                    | `mapFolding.basecamp.countFolds(mapShape, ...)`          |
| Count rotationally symmetric foldings          | `mapFolding.basecamp.countFoldsSymmetric(mapShape, ...)` |
| Count semi-meanders or meanders                | `mapFolding.basecamp.countMeanders(kind, n, ...)`        |
| Calculate any implemented OEIS term            | `mapFolding.oeis.oeisIDfor_n(oeisID, n, ...)`            |
| Convert a map-folding OEIS index to dimensions | `mapFolding.oeis.makeMapShape(oeisID, n)`                |
| Retrieve cached known values                   | `mapFolding.oeis.getValuesKnown(oeisID)`                 |
| List implemented sequences                     | `getOEISids`                                             |

`oeisIDfor_n` dispatches to the appropriate folding algorithm, meander algorithm, symmetric-folding algorithm, or exact formula. Run `getOEISids` for the current list and descriptions of all supported sequences.

The map-folding sequence mappings are:

| OEIS ID                             | Problem                               | `mapShape` for index `n` |
| ----------------------------------- | ------------------------------------- | ------------------------ |
| [A000136](https://oeis.org/A000136) | Strip of `n` labeled stamps           | `(1, n)`                 |
| [A001415](https://oeis.org/A001415) | `2 × n` strip                         | `(2, n)`                 |
| [A001416](https://oeis.org/A001416) | `3 × n` strip                         | `(3, n)`                 |
| [A001417](https://oeis.org/A001417) | `n`-dimensional `2 × ⋯ × 2` map       | `(2,) * n`               |
| [A195646](https://oeis.org/A195646) | `n`-dimensional `3 × ⋯ × 3` map       | `(3,) * n`               |
| [A001418](https://oeis.org/A001418) | `n × n` sheet                         | `(n, n)`                 |
| [A007822](https://oeis.org/A007822) | Symmetric foldings of `2n + 1` stamps | `(1, 2 * n)`             |

For A007822, pass `(1, 2 * n)` to `countFoldsSymmetric`; the function's computational shape differs from the `2n + 1` stamps in the sequence description.

OEIS metadata and b-files are cached locally for 30 days. Missing or stale entries are refreshed from `oeis.org`; stale cached data remains available if a refresh fails.

## Algorithm selection and long computations

Leave `flow=''` for the default implementation. The alternate selectors exist for research, validation, and performance comparisons:

| Interface             | Supported `flow` values                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| `countFolds`          | `''`, `daoOfMapFolding`, `numba`, `theorem2`, `theorem2Codon`, `theorem2Numba`, `theorem2Trimmed` |
| `countFoldsSymmetric` | `''`, `asynchronous`, `theorem2`, `theorem2Codon`, `theorem2Numba`, `theorem2Trimmed`             |
| `countMeanders`       | `''`, `matrixMeanders`, `matrixNumPy`, `matrixPandas`                                             |

The Numba, Pandas, and Codon selectors require their corresponding extras. For OEIS sequences with multiple exact identities, the `f` argument to `oeisIDfor_n` selects a formula; leaving it empty uses the default route.

Pass `pathLikeWrite` to a counting function to preserve a result. An existing directory receives a generated filename such as `p2x6.totalFolds`; an explicit target file is also supported. The destination is write-tested before computation, and an existing target is not overwritten.

`countFolds` can split work with an integer, `computationDivisions='cpu'`, or `computationDivisions='maximum'`. Dividing this algorithm repeats substantial work and is usually slower, so leave `computationDivisions=None` unless you are deliberately studying the parallel implementation.

## Repository guide

| Path                                                                                                                      | Role                                                                         |
| ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [`mapFolding/basecamp.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/basecamp.py)                    | Stable high-level dispatch for folding and meander computations              |
| [`mapFolding/oeis/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/oeis)                                 | OEIS dispatch, formulas, metadata, and cached values                         |
| [`mapFolding/algorithms/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/algorithms)                     | Handwritten source algorithms                                                |
| [`mapFolding/syntheticModules/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/syntheticModules)         | Generated implementations; regenerate these instead of editing them directly |
| [`mapFolding/someAssemblyRequired/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/someAssemblyRequired) | Project-specific AST transformations and module generators                   |
| [`mapFolding/_e/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/_e)                                     | Experimental elimination-based algorithms and analysis                       |
| [`mapFolding/tests/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/tests)                               | Main correctness, dispatch, filesystem, and parameter tests                  |
| [`mapFolding/reference/`](https://github.com/hunterhogan/mapFolding/tree/main/mapFolding/reference)                       | Historical implementations, completed jobs, notes, and research artifacts    |
| [`easyRun/`](https://github.com/hunterhogan/mapFolding/tree/main/easyRun)                                                 | Benchmark and exploration harnesses                                          |

General-purpose transformation primitives developed alongside this project now live in [astToolkit](https://github.com/hunterhogan/astToolkit) and [astToolFactory](https://github.com/hunterhogan/astToolFactory). The transformation pipeline retained here is specific to generating and validating `mapFolding` implementations.

## Development

Create and activate a virtual environment, then install both development extras:

```console
git clone https://github.com/hunterhogan/mapFolding.git
cd mapFolding
python -m venv .venv
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

POSIX shells:

```sh
source .venv/bin/activate
```

Install and test:

```console
pip install -e ".[development,testing]"
pytest
```

The test suite compares independent implementations with known OEIS values and stored data samples. When adding an algorithm variant, begin with [`mapFolding/tests/test_computations.py`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/tests/test_computations.py) and register the new flow beside the existing implementations.

## Citation

To cite this software, use the metadata in [`CITATION.cff`](https://github.com/hunterhogan/mapFolding/blob/main/CITATION.cff). BibTeX files for the mathematical literature are collected in [`citations/`](https://github.com/hunterhogan/mapFolding/tree/main/citations).

## Research references

### A map-folding problem

- [BibTeX citation.](https://github.com/hunterhogan/mapFolding/blob/main/citations/Lunnon1968.bib)
- DOI: [10.1090/S0025-5718-1968-0221957-8](https://doi.org/10.1090/S0025-5718-1968-0221957-8)
- PDF: [American Mathematical Society](https://pubs.ams.org/journals/mcom/1968-22-101/S0025-5718-1968-0221957-8/S0025-5718-1968-0221957-8.pdf)

### Folding a strip of stamps

- [BibTeX citation.](https://github.com/hunterhogan/mapFolding/blob/main/citations/Koehler1968.bib)
- DOI: [10.1016/S0021-9800(68)80048-1](https://doi.org/10.1016/S0021-9800(68)80048-1)

### Multi-dimensional map-folding

- [BibTeX citation.](https://github.com/hunterhogan/mapFolding/blob/main/citations/Lunnon.bib)
- DOI: [10.1093/comjnl/14.1.75](https://doi.org/10.1093/comjnl/14.1.75)
- PDF: [Oxford Academic](https://academic.oup.com/comjnl/article-pdf/14/1/75/1020149/140075.pdf)

### A transfer matrix approach to the enumeration of plane meanders

- [BibTeX citation.](https://github.com/hunterhogan/mapFolding/blob/main/citations/Jensen.bib)
- [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/cond-mat/0008178)
- DOI: [10.1088/0305-4470/33/34/301](https://doi.org/10.1088/0305-4470/33/34/301)
- Free preprint: [arXiv:cond-mat/0008178](https://arxiv.org/abs/cond-mat/0008178)

### A new transfer-matrix algorithm for exact enumerations: self-avoiding polygons on the square lattice

- [BibTeX citation.](https://github.com/hunterhogan/mapFolding/blob/main/citations/ClisbyJensen2012.bib)
- [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/1111.5877)
- DOI: [10.1088/1751-8113/45/11/115202](https://doi.org/10.1088/1751-8113/45/11/115202)
- Free preprint: [arXiv:1111.5877](https://arxiv.org/abs/1111.5877)

### Stamp Foldings, Semi-Meanders, and Open Meanders: Fast Generation Algorithms

- [BibTeX citation.](https://github.com/hunterhogan/mapFolding/blob/main/citations/Sawada2012.bib)
- DOI: [10.37236/2404](https://doi.org/10.37236/2404)
- PDF: [The Electronic Journal of Combinatorics](https://www.combinatorics.org/ojs/index.php/eljc/article/view/v19i2p43/pdf)

### Foldings and meanders

- [BibTeX citation.](https://github.com/hunterhogan/mapFolding/blob/main/citations/Legendre2014.bib)
- [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/1302.2025)
- PDF: [The Australasian Journal of Combinatorics](https://ajc.maths.uq.edu.au/pdf/58/ajc_v58_p275.pdf)
- Free preprint: [arXiv:1302.2025](https://arxiv.org/abs/1302.2025)

### Valid Orderings of Layers When Simple-Folding a Map

- [BibTeX citation.](https://github.com/hunterhogan/mapFolding/blob/main/citations/Jia2020.bib)
- DOI: [10.2197/ipsjjip.28.816](https://doi.org/10.2197/ipsjjip.28.816)
- PDF: [Journal of Information Processing](https://www.jstage.jst.go.jp/article/ipsjjip/28/0/28_816/_pdf/-char/en)

### jOEIS: Java Online Encyclopedia of Integer Sequences

- [BibTeX citation.](https://github.com/hunterhogan/mapFolding/blob/main/citations/jOEIS.bib)
- [Code repository.](https://github.com/archmageirvine/joeis)

### The Online Encyclopedia of Integer Sequences

- [BibTeX citation.](https://github.com/hunterhogan/mapFolding/blob/main/citations/oeis.bib)
- [Available at oeis.org.](https://oeis.org)

## Extending OEIS sequences

OEIS limits my proposed changes to a sequence to three sequences at a time. I have values to extend the following sequences, but I cannot submit them all at the same time.

- A060206, submitted.
- A077460
- A085973
- A208357
- A217310
- A217318, submitted.
- A223093
- A223094
- A223095, values.
- A333971, values.
- A334615, values.

## My recovery

[![2011 August: Homeless since](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)
[![YouTube channel subscribers](https://img.shields.io/youtube/channel/subscribers/UC3Gx7kz61009NbhpRtPP7tw)](https://www.youtube.com/@HunterHogan)

[![CC-BY-NC-4.0](https://raw.githubusercontent.com/hunterhogan/mapFolding/refs/heads/main/.github/CC-BY-NC-4.0.png)](https://creativecommons.org/licenses/by-nc/4.0/)
