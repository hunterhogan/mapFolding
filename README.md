# mapFolding: High-Performance Algorithm Playground for Computational Enthusiasts 🗺️

[![pip install mapFolding](https://img.shields.io/badge/pip%20install-mapFolding-gray.svg?colorB=3b434b)](https://pypi.org/project/mapFolding/)
[![Python Tests](https://github.com/hunterhogan/mapFolding/actions/workflows/pythonTests.yml/badge.svg)](https://github.com/hunterhogan/mapFolding/actions/workflows/pythonTests.yml)
[![License: CC-BY-NC-4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-3b434b)](https://creativecommons.org/licenses/by-nc/4.0/)

**This package is for you if:**

- You're fascinated by computational algorithms and their optimization
- You want to explore AST transformation techniques for Python performance tuning
- You're interested in solving mathematical puzzles through code
- You're learning about Numba and advanced Python optimization

## What Does This Package Actually Do?

`mapFolding` solves a specific mathematical problem: counting the number of distinct ways to fold a rectangular map. While this may sound niche, it's a fascinating computational challenge that demonstrates:

1. How to transform readable algorithms into blazingly fast implementations
2. Advanced techniques for Python optimization using AST manipulation
3. Numba acceleration with specialized compilation strategies
4. Algorithms for problems that grow combinatorially

The package has achieved new computational records, including first-ever calculations for large maps that were previously infeasible.

```python
# Compute the number of ways to fold a 5×5 grid:
from mapFolding import oeisIDfor_n
foldsTotal = oeisIDfor_n('A001418', 5)
```

## Key Benefits for Computational Enthusiasts

### 1. Algorithm Transformation Laboratory

See how the same algorithm evolves from readable Python to highly-optimized implementations:

```python
# The intuitive, readable version:
def countFolds(mapShape):
    # ...implement readable algorithm...

# The transformed, optimized version (auto-generated):
@numba.jit(nopython=True, parallel=True, fastmath=True)
def countFolds_optimized(shape_param):
    # ...blazingly fast implementation...
```

### 2. Code Generation Framework

Study and extend a complete Python code transformation assembly line:

- AST analysis and manipulation
- Dataclass decomposition ("shattering")
- Automatic import management
- Type specialization for numerical computing

### 3. Exhaustive Test Framework

Leverage a sophisticated test suite for validating your own optimizations:

```python
# Test your own recipe implementation with just a few lines:
@pytest.fixture
def myCustomRecipeFixture(useThisDispatcher, pathTmpTesting):
    myRecipe = RecipeSynthesizeFlow(
        # Your custom configuration here
    )
    # ...transformation code...
    return customDispatcher

def test_myCustomImplementation(myCustomRecipeFixture):
    # Automatic validation against known values
```

## Installation and Getting Started

```sh
pip install mapFolding
```

Try a quick calculation:

```python
from mapFolding import oeisIDfor_n

# Calculate ways to fold a 2×4 map
result = oeisIDfor_n('A001415', 4)  # Returns 8
print(f"A 2×4 map can be folded {result} different ways")
```

## Mathematical Background (For the Curious)

The map folding problem was introduced by Lunnon in 1971 and connects to combinatorial geometry, computational complexity, and integer sequence analysis. The calculations provide entries to the Online Encyclopedia of Integer Sequences (OEIS).

This package implements several OEIS sequences, including:

- A001415: Number of ways to fold a 2×n strip (now calculated up to n=20!)
- A001418: Number of ways to fold an n×n square grid

## Explore the Repository

The repository structure reveals the package's educational value:

- `reference/`: Historical implementations and algorithm evolution
- `someAssemblyRequired/`: Code transformation framework
- `tests/`: Comprehensive test suite with fixtures for your own implementations

## Who Is This For, Really?

If you've read this far and are intrigued by computational puzzles, algorithm optimization, or Python performance techniques, this package offers a playground for exploration. It's particularly valuable for:

- Computer science students studying algorithm optimization
- Python developers exploring Numba and AST manipulation
- Computational mathematicians interested in combinatorial problems
- Anyone fascinated by the intersection of mathematics and computing

Whether you use it to solve map folding problems or to study its optimization techniques, `mapFolding` offers a unique window into advanced Python programming approaches.

## My recovery

[![Static Badge](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC3Gx7kz61009NbhpRtPP7tw)](https://www.youtube.com/@HunterHogan)

## How to code

Coding One Step at a Time:

0. WRITE CODE.
1. Don't write stupid code that's hard to revise.
2. Write good code.
3. When revising, write better code.

[![CC-BY-NC-4.0](https://github.com/hunterhogan/mapFolding/blob/main/CC-BY-NC-4.0.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
