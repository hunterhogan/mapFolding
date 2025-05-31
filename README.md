# mapFolding

[![pip install mapFolding](https://img.shields.io/badge/pip%20install-mapFolding-gray.svg?colorB=3b434b)](https://pypi.org/project/mapFolding/)
[![Python Tests](https://github.com/hunterhogan/mapFolding/actions/workflows/pythonTests.yml/badge.svg)](https://github.com/hunterhogan/mapFolding/actions/workflows/pythonTests.yml)
[![License: CC-BY-NC-4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-3b434b)](https://creativecommons.org/licenses/by-nc/4.0/)

A computational framework that starts with Lunnon's 1971 algorithm for counting distinct ways to fold maps and improves it. Plus there is a comprehensive AST transformation system for transforming algorithms for optimization and research.

(Yo, the rest is AI generated and I don't have the energy to proofread it. This package helped me compute two previously unknown values: I'm sure others can improve it.)

## The Mathematical Problem

Map folding is a combinatorial problem: given a rectangular grid of unit squares, how many distinct ways can you fold it? "Distinct" means that foldings producing identical final shapes are counted as one. This problem connects to combinatorial geometry, integer sequences, and computational complexity theory.

The calculations extend the Online Encyclopedia of Integer Sequences (OEIS):

- **A001415**: 2×n strips (computed through n=20 for the first time)
- **A001418**: n×n squares
- **A001416**: 3×n strips
- **A001417**: n-dimensional hypercubes
- **A195646**: 3×3×...×3 hypercubes

```python
from mapFolding import oeisIDfor_n

# How many ways can you fold a 2×4 strip?
foldsTotal = oeisIDfor_n('A001415', 4)
```

## The Computational Challenge

For larger maps, these calculations require hours or days to complete. A 2×20 strip requires processing leaves through billions of recursive operations. The package addresses this through systematic algorithm transformation: converting readable Python implementations into specialized, Numba-optimized modules that achieve order-of-magnitude performance improvements.

## What This Package Provides

### Core Functionality

- **Complete implementation** of Lunnon's recursive algorithm
- **Mathematical validation** through OEIS integration and caching
- **Type-safe computational state** management with automatic initialization
- **Result persistence** for long-running calculations

### Algorithm Transformation System

- **AST manipulation framework** for converting dataclass-based algorithms to optimized implementations
- **Automatic code generation** that produces standalone, highly optimized computation modules
- **Dataclass decomposition** to enable Numba compatibility while preserving readable source code
- **Comprehensive optimization** including dead code elimination, static value embedding, and aggressive compilation settings

### Educational Resources

- **Historical implementations** showing algorithm evolution from 1971 to present
- **Performance comparison** studies demonstrating optimization techniques
- **Complete test suite** with patterns for validating custom implementations
- **Reference documentation** for extending the transformation framework

## Use Cases

**Mathematical Research**: Explore folding pattern properties, extend known sequences, or validate theoretical results against computed values.

**Algorithm Optimization Learning**: Study a complete transformation pipeline that converts high-level algorithms into production-ready optimized code.

**Performance Computing Education**: Examine techniques for achieving maximum Python performance through Numba integration, AST manipulation, and specialized code generation.

**Combinatorial Problem Solving**: Use the framework as a template for optimizing other recursive combinatorial algorithms.

## Example Usage

```python
from mapFolding import countFolds

# Count folding patterns for a 3×3 square
result = countFolds([3, 3])

# Access OEIS sequences directly
from mapFolding import oeisIDfor_n
strip_foldings = oeisIDfor_n('A001415', 6)  # 2×6 strip

# Generate optimized code for specific dimensions
from mapFolding.someAssemblyRequired import makeJobTheorem2Numba
# Creates specialized modules for maximum performance
```

## Repository Structure

- `mapFolding/`: Core implementation with modular architecture
- `reference/`: Historical algorithm implementations and performance studies
- `someAssemblyRequired/`: AST transformation framework
- `tests/`: Comprehensive validation suite
- `jobs/`: Generated optimized modules for specific calculations

## Performance Characteristics

- **Pure Python baseline**: Educational implementations for understanding
- **NumPy optimization**: ~10× improvement through vectorized operations
- **Numba compilation**: ~100× improvement through native code generation
- **Specialized modules**: ~1000× improvement through static optimization and embedded constants

Actual performance varies by map dimensions and available hardware.

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
