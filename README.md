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
  A001415: Number of ways of folding a 2 X n strip of stamps.
  A001416: Number of ways of folding a 3 X n strip of stamps.
  A001417: Number of ways of folding a 2 X 2 X ... X 2 n-dimensional map.
  A001418: Number of ways of folding an n X n sheet of stamps.
  A195646: Number of ways of folding a 3 X 3 X ... X 3 n-dimensional map.
```

### 2. **Algorithm Zoo** 🦒

- **Lunnon's 1971 Algorithm**: A painstakingly debugged version of [the original typo-riddled code](https://github.com/hunterhogan/mapFolding/blob/mapFolding/reference/foldings.txt)
- The /reference directory.
- **Numba-JIT Accelerated**: Up to 1000× faster than pure Python ([benchmarks](https://github.com/hunterhogan/mapFolding/blob/mapFolding/notes/Speed%20highlights.md))

### 3. **For Researchers and Power Users** 🔬

This package provides a sophisticated code transformation framework that can turn readable algorithm implementations into highly-optimized computational engines:

- **Algorithmic Exploration**: Study the core algorithm in `theDao.py`, which uses a functional state-transformation approach with clear, isolated functions
- **Performance Optimization**: Generate specialized implementations with the `someAssemblyRequired` transformation pipeline:
  - AST-based code analysis and manipulation
  - Dataclass "shattering" to decompose complex state objects into primitive components
  - Just-in-time compilation with Numba and various optimization profiles
  - LLVM IR extraction for low-level algorithmic analysis

- **Extensible Design**: The transformation framework is abstract and generic, enabling:
  - Creation of new optimization targets beyond the included Numba implementation
  - Customization of compilation parameters and optimization levels
  - Development of specialized algorithms for specific map dimensions

### 4. **Customization and Extension Guide**

The package architecture supports multiple levels of customization:

- **Basic Usage**: Work with the high-level API in `basecamp.py` for standard computations
- **Algorithm Modification**:
  - Modify the core algorithm in `theDao.py` while preserving its functional approach
  - Configure system-wide settings in `theSSOT.py` to adjust data types and performance characteristics
  - Use utility functions in `beDRY.py` for common operations

- **Advanced Transformation**:
  - The `someAssemblyRequired` package provides tools to transform code at the AST level:
    - `transformationTools.py` contains utilities for AST manipulation and code generation
    - `transformDataStructures.py` handles complex data structure transformations
    - `ingredientsNumba.py` provides Numba-specific configuration profiles
    - `synthesizeNumbaFlow.py` orchestrates the transformation process

- **Custom Deployment**:
  - Generate specialized implementations for specific dimensions
  - Create optimized modules that can be executed as standalone scripts
  - Extract LLVM IR for further analysis or optimization

The package's multi-level design allows you to start with simple API calls and progressively delve deeper into optimization as your computational needs grow.

## Map-folding Video

~~This caused my neurosis:~~ I enjoyed the following video, which is what introduced me to map folding.

"How Many Ways Can You Fold a Map?" by Physics for the Birds, 2024 November 13 ([BibTex](https://github.com/hunterhogan/mapFolding/blob/main/citations/Physics_for_the_Birds.bibtex) citation)

[![How Many Ways Can You Fold a Map?](https://i.ytimg.com/vi/sfH9uIY3ln4/hq720.jpg)](https://www.youtube.com/watch?v=sfH9uIY3ln4)

---

## My recovery

[![Static Badge](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC3Gx7kz61009NbhpRtPP7tw)](https://www.youtube.com/@HunterHogan)

[![CC-BY-NC-4.0](https://github.com/hunterhogan/mapFolding/blob/main/CC-BY-NC-4.0.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
