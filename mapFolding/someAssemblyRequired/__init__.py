"""
Map folding AST transformation system: Comprehensive framework for converting dataclass-based algorithms to optimized implementations.

This subpackage implements a sophisticated Abstract Syntax Tree (AST) transformation system specifically designed to convert
high-level dataclass-based map folding algorithms into highly optimized, Numba-compatible implementations. The transformation
system addresses a fundamental challenge in high-performance scientific computing: bridging the gap between maintainable,
object-oriented algorithm implementations and the performance requirements of computationally intensive mathematical research.

The map folding problem domain involves complex combinatorial calculations that can require hours or days to complete for specific
dimensional configurations. While dataclass-based implementations provide clean, maintainable interfaces for managing complex
mathematical state, these objects cannot be directly processed by Numba's just-in-time compiler, which excels at optimizing
operations on primitive values and tuples. This subpackage resolves this architectural tension through systematic AST manipulation
that preserves algorithmic correctness while enabling dramatic performance improvements.

## System Architecture

The transformation system operates through a carefully orchestrated sequence of specialized modules, each contributing essential
capabilities to the complete transformation process:

### Foundation Layer: Pattern Recognition and Structural Analysis
- `_toolIfThis`: Extended predicate functions for identifying specific code patterns in AST nodes, particularly conditional
	expressions and control flow structures essential to map folding computations
- `_toolkitContainers`: Dataclass decomposition containers that extract individual fields, type annotations, and reconstruction
	logic from dataclass definitions into manipulatable AST components

### Operational Core: Transformation Implementation
- `transformationTools`: Core functions executing dataclass decomposition, function signature transformation, and calling
	convention adaptation that convert dataclass-accepting functions into primitive-parameter equivalents
- `toolkitNumba`: Numba integration tools providing just-in-time compilation optimization with configurable performance parameters
	and strategic compiler directive application

### Configuration and Orchestration
- `infoBooth`: Configuration constants, computational complexity estimates, and default identifiers for systematic module generation and optimization decision-making
- `RecipeJob`: Configuration management dataclasses that coordinate transformation parameters across multiple stages while
	maintaining consistency between source algorithms and target optimizations
- `makeAllModules`: Comprehensive transformation orchestration tools that execute complete transformation processes for diverse
	computational strategies and performance characteristics
- `makeJobTheorem2Numba`: Specialized job generation implementing the complete transformation sequence to produce standalone, highly optimized computation modules

### Utility Extensions
- `getLLVMforNoReason`: Standalone utility for extracting LLVM Intermediate Representation from compiled modules for debugging and optimization analysis

## Transformation Process

The complete transformation follows a systematic three-stage pattern:

1. **Analysis and Decomposition**: Pattern recognition identifies dataclass structures and dependencies, followed by decomposition
	into constituent AST components including field definitions, type annotations, and initialization patterns.

2. **Function Optimization**: Core transformations convert functions accepting dataclass parameters into functions accepting
	individual primitive parameters, with systematic updates to signatures, return types, and calling conventions.

3. **Compilation Integration**: Numba decorators with carefully configured optimization parameters are applied to transformed
	functions, enabling aggressive just-in-time compilation with performance characteristics suitable for large-scale computational
	research.

## Generated Module Characteristics

The transformation system produces standalone Python modules with embedded constants replacing parameterized values, eliminated
dead code paths, optimized data structures, Numba compilation directives, progress feedback for long-running calculations, and
consistent naming conventions with systematic filesystem organization. These modules maintain mathematical correctness while
providing significant performance improvements essential to map folding research computational demands.

## Usage Guidance

Begin exploration with `infoBooth` for understanding configuration options and complexity estimates. Proceed to
`transformationTools` for core transformation capabilities, then examine `RecipeJob` for orchestration patterns. Advanced users
developing custom transformations should study `_toolIfThis` and `_toolkitContainers` for foundational pattern recognition and
structural manipulation capabilities.

The transformation system represents the culmination of systematic AST manipulation research, enabling previously intractable
calculations through the strategic application of compiler optimization techniques to abstract mathematical algorithms.
"""

from mapFolding.someAssemblyRequired.infoBooth import (
	dataclassInstanceIdentifierDEFAULT as dataclassInstanceIdentifierDEFAULT,
	sourceCallableDispatcherDEFAULT as sourceCallableDispatcherDEFAULT,
)

from mapFolding.someAssemblyRequired._toolIfThis import IfThis as IfThis

from mapFolding.someAssemblyRequired._toolkitContainers import (
	DeReConstructField2ast as DeReConstructField2ast,
	ShatteredDataclass as ShatteredDataclass,
)
