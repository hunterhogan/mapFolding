# mapFolding: Algorithms for enumerating distinct map/stamp folding patterns 🗺️

[![pip install mapFolding](https://img.shields.io/badge/pip%20install-mapFolding-gray.svg?colorB=3b434b)](https://pypi.org/project/mapFolding/)
[![Static Badge](https://img.shields.io/badge/stinkin'%20badges-don't%20need-b98e5e)](https://youtu.be/g6f_miE91mk&t=4)
[![Python Tests](https://github.com/hunterhogan/mapFolding/actions/workflows/pythonTests.yml/badge.svg)](https://github.com/hunterhogan/mapFolding/actions/workflows/pythonTests.yml)
![Static Badge](https://img.shields.io/badge/issues-I%20have%20them-brightgreen)
[![License: CC-BY-NC-4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-3b434b)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Quick start

```sh
pip install mapFolding
```

`OEIS_for_n` will run a computation from the command line.

```cmd
(mapFolding) C:\apps\mapFolding> OEIS_for_n A001418 5
186086600 distinct folding patterns.
Time elapsed: 1.605 seconds
```

Use `mapFolding.oeisIDfor_n()` to compute a(n) for an OEIS ID.

```python
from mapFolding import oeisIDfor_n
foldsTotal = oeisIDfor_n( 'A001418', 4 )
```

---

## Features

### 1. Simple, easy usage based on OEIS IDs

`mapFolding` directly implements some IDs from [_The On-Line Encyclopedia of Integer Sequences_](https://oeis.org/) ([BibTex](https://github.com/hunterhogan/mapFolding/blob/main/citations/oeis.bibtex) citation).

Use `getOEISids` to get the most up-to-date list of available OEIS IDs.

```cmd
(mapFolding) C:\apps\mapFolding> getOEISids

Available OEIS sequences:
  A001415: Number of ways of folding a 2 X n strip of stamps. (Now extended to n=20!)
  A001416: Number of ways of folding a 3 X n strip of stamps.
  A001417: Number of ways of folding a 2 X 2 X ... X 2 n-dimensional map.
  A001418: Number of ways of folding an n X n sheet of stamps.
  A195646: Number of ways of folding a 3 X 3 X ... X 3 n-dimensional map.
```

### 2. **Algorithm Zoo: A Historical and Performance Journey** 🦒

This package offers a comprehensive collection of map folding algorithm implementations that showcase its evolution from historical origins to high-performance computation:

- **Historical Implementations**:
  - Carefully restored versions of Lunnon's 1971 original [algorithm](https://github.com/hunterhogan/mapFolding/blob/mapFolding/reference/foldings.txt) with corrections
  - Atlas Autocode reconstruction in the `reference/foldings.AA` file

- **Direct Translations**:
  - Python translations following the original control flow (`lunnanWhile.py`)
  - NumPy-based vectorized implementations (`lunnanNumpy.py`)

- **Modern Implementations**:
  - Java port adaptations (`irvineJavaPort.py`) providing cleaner procedural implementations
  - Experimental JAX version (`jaxCount.py`) exploring GPU acceleration potential
  - Semantically decomposed version (`flattened.py`) with clear function boundaries

- **Performance Optimized**:
  - Numba-JIT accelerated implementations up to 1000× faster than pure Python (see [benchmarks](https://github.com/hunterhogan/mapFolding/blob/mapFolding/notes/Speed%20highlights.md))
  - Algorithmic optimizations showcasing subtle yet powerful performance differences (`total_countPlus1vsPlusN.py`)
  - **New Computations**: First-ever calculations for 2×19 and 2×20 maps in the `reference/jobsCompleted/` directory

The `reference` directory serves as both a historical archive and an educational resource for understanding algorithm evolution.

### 3. **Algorithmic Transformation: From Readability to Speed** 🔬

The package provides a sophisticated transformation framework that bridges the gap between human-readable algorithms and high-performance computation:

- **Core Algorithm Understanding**:
  - Study the functional state-transformation approach in `theDao.py` with clear, isolated functions
  - Explore the semantic decomposition in `reference/flattened.py` to understand algorithm sections

- **Code Transformation Assembly-line**:
  - **AST Manipulation**: Analyzes and transforms the algorithm's abstract syntax tree
  - **Dataclass "Shattering"**: Decomposes complex state objects into primitive components
  - **Optimization Applications**: Applies domain-specific optimizations for numerical computation
  - **LLVM Integration**: Extracts LLVM IR for low-level algorithmic analysis

- **Performance Breakthroughs**:
  - Learn why nearly identical algorithms can have dramatically different performance (`total_countPlus1vsPlusN.py`)
  - See how memory layout and increment strategy impact computation speed
  - Understand the batching technique that yields order-of-magnitude improvements

### 4. **Multi-Level Architecture: From Simple API to Full Customization**

The package's architecture supports multiple levels of engagement:

- **Basic Usage**:
  - Work with the high-level API in `basecamp.py` for standard computations
  - Access OEIS sequence calculations with minimal code

- **Algorithm Exploration**:
  - Compare different implementations in the `reference` directory to understand trade-offs
  - Modify the core algorithm in `theDao.py` while preserving its functional approach
  - Configure system-wide settings in `theSSOT.py` to adjust data types and performance characteristics

- **Advanced Transformation**:
  - Use the `someAssemblyRequired` package to transform algorithms at the AST level
  - Create optimized variants with different compilation settings using:
    - `transformationTools.py` for AST manipulation
    - `transformDataStructures.py` for complex data structure transformations
    - `ingredientsNumba.py` for Numba-specific optimization profiles
    - `synthesizeNumbaFlow.py` to orchestrate the transformation process

- **Custom Deployment**:
  - Generate specialized implementations for specific dimensions
  - Create optimized standalone modules for production use
  - Extract LLVM IR for further analysis and optimization

The package's multi-level design allows you to start with simple API calls and progressively explore deeper optimization techniques as your computational needs grow.

## Map-folding Video

~~This caused my neurosis:~~ I enjoyed the following video, which is what introduced me to map folding.

"How Many Ways Can You Fold a Map?" by Physics for the Birds, 2024 November 13 ([BibTex](https://github.com/hunterhogan/mapFolding/blob/main/citations/Physics_for_the_Birds.bibtex) citation)

[![How Many Ways Can You Fold a Map?](https://i.ytimg.com/vi/sfH9uIY3ln4/hq720.jpg)](https://www.youtube.com/watch?v=sfH9uIY3ln4)

---

## My recovery

[![Static Badge](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC3Gx7kz61009NbhpRtPP7tw)](https://www.youtube.com/@HunterHogan)

[![CC-BY-NC-4.0](https://github.com/hunterhogan/mapFolding/blob/main/CC-BY-NC-4.0.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
