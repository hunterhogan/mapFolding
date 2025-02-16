# mapFolding: Algorithms for enumerating distinct map/stamp folding patterns 🗺️

[![pip install mapFolding](https://img.shields.io/badge/pip%20install-mapFolding-gray.svg?colorB=3b434b)](https://pypi.org/project/mapFolding/)
[![Static Badge](https://img.shields.io/badge/stinkin'%20badges-don't%20need-b98e5e)](https://youtu.be/g6f_miE91mk&t=4)
[![Python Tests](https://github.com/hunterhogan/mapFolding/actions/workflows/pythonTests.yml/badge.svg)](https://github.com/hunterhogan/mapFolding/actions/workflows/pythonTests.yml)
![Static Badge](https://img.shields.io/badge/issues-I%20have%20them-brightgreen)
[![License: CC-BY-NC-4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-3b434b)](https://creativecommons.org/licenses/by-nc/4.0/)
![PyPI - Downloads](https://img.shields.io/pypi/dd/mapFolding)
![GitHub repo size](https://img.shields.io/github/repo-size/hunterhogan/mapFolding)

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

- **Lunnon’s 1971 Algorithm**: A painstakingly debugged version of [the original typo-riddled code](https://github.com/hunterhogan/mapFolding/blob/mapFolding/reference/foldings.txt)
- The /reference directory.
- **Numba-JIT Accelerated**: Up to 1000× faster than pure Python ([benchmarks](https://github.com/hunterhogan/mapFolding/blob/mapFolding/notes/Speed%20highlights.md))

### 3. **For Researchers** 🔬

- Change multiple minute settings, such as the bit width of the data types.
- Transform the algorithm using AST
- Create hyper-optimized modules to compute a specific map.

### 4. **Customizing your algorithm**

- mapFolding\someAssemblyRequired\synthesizeNumbaJob.py (and/or synthesizeNumba____.py, as applicable)
  - Synthesize a Numba-optimized module for a specific mapShape
  - Synthesize _from_ a module in mapFolding\syntheticModules or from any source you select
  - Use the existing transformation options
  - Or create new ways of transforming the algorithm from its source to a specific job
- mapFolding\someAssemblyRequired\makeJob.py
  - Initialize data for a specific mapShape
- mapFolding\someAssemblyRequired\synthesizeNumbaModules.py (and/or synthesizeNumba____.py, as applicable)
  - Synthesize one or more Numba-optimized modules for parallel or sequential computation
  - Overwrite the modules in mapFolding\syntheticModules or save the module(s) to a custom path
  - Synthesize _from_ the algorithm(s) in mapFolding\theDao.py or from any source you select
  - Use the existing transformation options
  - Or create new ways of transforming the algorithm from its source to a new module
  - Use your new module in synthesizeNumbaJob.py, above, as the source to create a mapShape-specific job module
- mapFolding\theDao.py
  - Modify the algorithms for initializing values, parallel computation, and/or sequential computation
  - Use the modified algorithm(s) in synthesizeNumbaModules.py, above, to create Numba-optimized version(s)
  - Then use a Numba-optimized version in synthesizeNumbaJob.py, above, to create a hyper-optimized version for a specific mapShape
- mapFolding\theSSOT.py (and/or theSSOTnumba.py and/ or theSSOT____.py, if they exist)
  - Modify broad settings or find functions to modify broad settings, such as data structures and their data types
  - Create new settings or groups of settings
- mapFolding\beDRY.py
  - Functions to handle common tasks, such as parsing parameters or creating the `connectionGraph` for a mapShape (a Cartesian product decomposition)
- mapFolding\someAssemblyRequired
  - Create new transformations to optimize the algorithm, such as for JAX, CuPy, or CUDA
  - (mapFolding\reference\jax.py has a once-functional JAX implementation, and synthesizeModuleJAX.py might be a useful starting point)

## Map-folding Video

~~This caused my neurosis:~~ I enjoyed the following video, which is what introduced me to map folding.

"How Many Ways Can You Fold a Map?" by Physics for the Birds, 2024 November 13 ([BibTex](https://github.com/hunterhogan/mapFolding/blob/main/citations/Physics_for_the_Birds.bibtex) citation)

[![How Many Ways Can You Fold a Map?](https://i.ytimg.com/vi/sfH9uIY3ln4/hq720.jpg)](https://www.youtube.com/watch?v=sfH9uIY3ln4)

---

## My recovery

[![Static Badge](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC3Gx7kz61009NbhpRtPP7tw)](https://www.youtube.com/@HunterHogan)

[![CC-BY-NC-4.0](https://github.com/hunterhogan/mapFolding/blob/main/CC-BY-NC-4.0.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
