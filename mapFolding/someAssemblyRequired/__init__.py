"""
Code Transformation Framework for Algorithm Optimization and Testing

This package implements a comprehensive framework for programmatically analyzing, transforming, and generating optimized
Python code. It serves as the algorithmic optimization engine for the mapFolding package, enabling the conversion of
readable, functional implementations into highly-optimized variants with verified correctness.

## Core Architecture Components

1. **AST Manipulation Tools**
	- Pattern recognition with composable predicates (ifThis)
	- Node access with consistent interfaces (DOT)
	- AST traversal and transformation (NodeChanger, NodeTourist)
	- AST construction with sane defaults (Make)
	- Node transformation operations (grab, Then)

2. **Container and Organization**
	- Import tracking and management (LedgerOfImports)
	- Function packaging with dependencies (IngredientsFunction)
	- Module assembly with structured components (IngredientsModule)
	- Recipe configuration for generating optimized code (RecipeSynthesizeFlow)
	- Dataclass decomposition for compatibility (ShatteredDataclass)

3. **Optimization assembly lines**
	- General-purpose Numba acceleration (makeNumbaFlow)
	- Job-specific optimization for concrete parameters (makeJobNumba)
	- Specialized component transformation (decorateCallableWithNumba)

## Integration with Testing Framework

The transformation components are extensively tested through the package's test suite, which provides specialized
fixtures and utilities for validating both the transformation process and the resulting optimized code:

- **syntheticDispatcherFixture**: Creates and tests a complete Numba-optimized module using RecipeSynthesizeFlow
	configuration

- **test_writeJobNumba**: Tests the job-specific optimization process with RecipeJob

These fixtures enable users to test their own custom recipes and job configurations with minimal effort. See the
documentation in tests/__init__.py for details on extending the test suite for custom implementations.

The framework balances multiple optimization levels - from general algorithmic improvements to parameter-specific
optimizations - while maintaining the ability to verify correctness at each transformation stage through the integrated
test suite.
"""

from astToolkit import (
	ast_expr_Slice as ast_expr_Slice,
	ast_Identifier as ast_Identifier,
	astModuleToIngredientsFunction as astModuleToIngredientsFunction,
	Be as Be,
	DOT as DOT,
	extractClassDef as extractClassDef,
	extractFunctionDef as extractFunctionDef,
	Grab as Grab,
	IngredientsFunction as IngredientsFunction,
	IngredientsModule as IngredientsModule,
	LedgerOfImports as LedgerOfImports,
	Make as Make,
	NodeChanger as NodeChanger,
	NodeORattribute as NodeORattribute,
	NodeTourist as NodeTourist,
	parseLogicalPath2astModule as parseLogicalPath2astModule,
	parsePathFilename2astModule as parsePathFilename2astModule,
	str_nameDOTname as str_nameDOTname,
	Then as Then,
)

from Z0Z_tools import importLogicalPath2Callable as importLogicalPath2Callable, importPathFilename2Callable as importPathFilename2Callable

from astToolkit.transformationTools import (
	inlineFunctionDef as inlineFunctionDef,
	removeUnusedParameters as removeUnusedParameters,
	unparseFindReplace as unparseFindReplace,
	write_astModule as write_astModule,
)

from mapFolding.someAssemblyRequired._toolIfThis import IfThis as IfThis

from mapFolding.someAssemblyRequired._toolkitContainers import (
	DeReConstructField2ast as DeReConstructField2ast,
	RecipeSynthesizeFlow as RecipeSynthesizeFlow,
	ShatteredDataclass as ShatteredDataclass,
)
