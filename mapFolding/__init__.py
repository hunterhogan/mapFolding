"""
Map folding enumeration and counting algorithms with advanced optimization capabilities.

This package implements algorithms to count and enumerate the distinct ways a rectangular map can be folded, based on
the mathematical problem described in Lunnon's 1971 paper. It provides multiple layers of functionality, from high-level
user interfaces to sophisticated algorithmic optimizations and code transformation tools.

Core modules:
- basecamp: Public API with simplified interfaces for end users
- theDao: Core computational algorithm using a functional state-transformation approach
- beDRY: Core utility functions implementing consistent data handling, validation, and resource management across the
	package's computational assembly-line
- theSSOT: Single Source of Truth for configuration, types, and state management
- toolboxFilesystem: Cross-platform file management services for storing and retrieving computation results with robust
	error handling and fallback mechanisms
- oeis: Interface to the Online Encyclopedia of Integer Sequences for known results

Extended functionality:
- someAssemblyRequired: Code transformation framework that optimizes the core algorithm through AST manipulation,
	dataclass transformation, and compilation techniques
	- The system converts readable code into high-performance implementations through a systematic analysis and
		transformation assembly line
	- Provides tools to "shatter" complex dataclasses into primitive components, enabling compatibility with Numba and
		other optimization frameworks
	- Creates specialized implementations tailored for specific input parameters

Testing and extension:
- tests: Comprehensive test suite designed for both verification and extension
	- Provides fixtures and utilities that simplify testing of custom implementations
	- Enables users to validate their own recipes and job configurations with minimal code
	- Offers standardized testing patterns that maintain consistency across the codebase
	- See tests/__init__.py for detailed documentation on extending the test suite

Special directories:
- .cache/: Stores cached data from external sources like OEIS to improve performance
- syntheticModules/: Contains dynamically generated, optimized implementations of the core algorithm created by the code
transformation framework
- reference/: Historical implementations and educational resources for algorithm exploration
	- reference/jobsCompleted/: Contains successful computations for previously unknown values, including first-ever
		calculations for 2x19 and 2x20 maps (OEIS A001415)

This package balances algorithm readability and understandability with high-performance computation capabilities,
allowing users to compute map folding totals for larger dimensions than previously feasible while also providing a
foundation for exploring advanced code transformation techniques.
"""

from typing import Any, TypeAlias
import sys

stuPyd: TypeAlias = Any

if sys.version_info >= (3, 12):
	from ast import (
		ParamSpec as astDOTParamSpec,
		type_param as astDOTtype_param,
		TypeAlias as astDOTTypeAlias,
		TypeVar as astDOTTypeVar,
		TypeVarTuple as astDOTTypeVarTuple,
	)
else:
	astDOTParamSpec: TypeAlias = stuPyd
	astDOTtype_param: TypeAlias = stuPyd
	astDOTTypeAlias: TypeAlias = stuPyd
	astDOTTypeVar: TypeAlias = stuPyd
	astDOTTypeVarTuple: TypeAlias = stuPyd

if sys.version_info >= (3, 11):
	from ast import TryStar as astDOTTryStar
	from typing import TypedDict as TypedDict
	from typing import NotRequired as NotRequired
else:
	astDOTTryStar: TypeAlias = stuPyd
	try:
		from typing_extensions import TypedDict as TypedDict
		from typing_extensions import NotRequired as NotRequired
	except Exception:
		TypedDict = dict[stuPyd, stuPyd]
		from collections.abc import Iterable
		NotRequired: TypeAlias = Iterable

from mapFolding.datatypes import (  # noqa: E402
	Array1DElephino as Array1DElephino,
	Array1DFoldsTotal as Array1DFoldsTotal,
	Array1DLeavesTotal as Array1DLeavesTotal,
	Array3D as Array3D,
	DatatypeElephino as DatatypeElephino,
	DatatypeFoldsTotal as DatatypeFoldsTotal,
	DatatypeLeavesTotal as DatatypeLeavesTotal,
	NumPyElephino as NumPyElephino,
	NumPyFoldsTotal as NumPyFoldsTotal,
	NumPyIntegerType as NumPyIntegerType,
	NumPyLeavesTotal as NumPyLeavesTotal,
)

from mapFolding.theSSOT import (  # noqa: E402
	ComputationState as ComputationState,
	raiseIfNoneGitHubIssueNumber3 as raiseIfNoneGitHubIssueNumber3,
	The as The,
)

from mapFolding.theDao import (  # noqa: E402
	countInitialize as countInitialize,
	doTheNeedful as doTheNeedful,
)

from mapFolding.beDRY import (  # noqa: E402
	getLeavesTotal as getLeavesTotal,
	getTaskDivisions as getTaskDivisions,
	outfitCountFolds as outfitCountFolds,
	setProcessorLimit as setProcessorLimit,
	validateListDimensions as validateListDimensions,
)

from mapFolding.toolboxFilesystem import (  # noqa: E402
	getPathFilenameFoldsTotal as getPathFilenameFoldsTotal,
	getPathRootJobDEFAULT as getPathRootJobDEFAULT,
	saveFoldsTotal as saveFoldsTotal,
	saveFoldsTotalFAILearly as saveFoldsTotalFAILearly,
	writeStringToHere as writeStringToHere,
)

from mapFolding.basecamp import countFolds as countFolds  # noqa: E402

from mapFolding.oeis import (  # noqa: E402
	clearOEIScache as clearOEIScache,
	getFoldsTotalKnown as getFoldsTotalKnown,
	getOEISids as getOEISids,
	OEIS_for_n as OEIS_for_n,
	oeisIDfor_n as oeisIDfor_n,
)
