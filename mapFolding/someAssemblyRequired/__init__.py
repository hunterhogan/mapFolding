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

from mapFolding.someAssemblyRequired._astTypes import (
	hasDOTannotation_expr as hasDOTannotation_expr,
	hasDOTannotation_exprORNone as hasDOTannotation_exprORNone,
	hasDOTannotation as hasDOTannotation,
	hasDOTarg_str as hasDOTarg_str,
	hasDOTarg_strORNone as hasDOTarg_strORNone,
	hasDOTarg as hasDOTarg,
	hasDOTargs_arguments as hasDOTargs_arguments,
	hasDOTargs_list_arg as hasDOTargs_list_arg,
	hasDOTargs_list_expr as hasDOTargs_list_expr,
	hasDOTargs as hasDOTargs,
	hasDOTargtypes as hasDOTargtypes,
	hasDOTasname as hasDOTasname,
	hasDOTattr as hasDOTattr,
	hasDOTbases as hasDOTbases,
	hasDOTbody_expr as hasDOTbody_expr,
	hasDOTbody_list_stmt as hasDOTbody_list_stmt,
	hasDOTbody as hasDOTbody,
	hasDOTbound as hasDOTbound,
	hasDOTcases as hasDOTcases,
	hasDOTcause as hasDOTcause,
	hasDOTcls as hasDOTcls,
	hasDOTcomparators as hasDOTcomparators,
	hasDOTcontext_expr as hasDOTcontext_expr,
	hasDOTconversion as hasDOTconversion,
	hasDOTctx as hasDOTctx,
	hasDOTdecorator_list as hasDOTdecorator_list,
	hasDOTdefault_value as hasDOTdefault_value,
	hasDOTdefaults as hasDOTdefaults,
	hasDOTelt as hasDOTelt,
	hasDOTelts as hasDOTelts,
	hasDOTexc as hasDOTexc,
	hasDOTfinalbody as hasDOTfinalbody,
	hasDOTformat_spec as hasDOTformat_spec,
	hasDOTfunc as hasDOTfunc,
	hasDOTgenerators as hasDOTgenerators,
	hasDOTguard as hasDOTguard,
	hasDOThandlers as hasDOThandlers,
	hasDOTid as hasDOTid,
	hasDOTifs as hasDOTifs,
	hasDOTis_async as hasDOTis_async,
	hasDOTitems as hasDOTitems,
	hasDOTiter as hasDOTiter,
	hasDOTkey as hasDOTkey,
	hasDOTkeys as hasDOTkeys,
	hasDOTkeywords as hasDOTkeywords,
	hasDOTkind as hasDOTkind,
	hasDOTkw_defaults as hasDOTkw_defaults,
	hasDOTkwarg as hasDOTkwarg,
	hasDOTkwd_attrs as hasDOTkwd_attrs,
	hasDOTkwd_patterns as hasDOTkwd_patterns,
	hasDOTkwonlyargs as hasDOTkwonlyargs,
	hasDOTleft as hasDOTleft,
	hasDOTlevel as hasDOTlevel,
	hasDOTlineno as hasDOTlineno,
	hasDOTlower as hasDOTlower,
	hasDOTmodule as hasDOTmodule,
	hasDOTmsg as hasDOTmsg,
	hasDOTname_expr as hasDOTname_expr,
	hasDOTname_str as hasDOTname_str,
	hasDOTname_strORNone as hasDOTname_strORNone,
	hasDOTname as hasDOTname,
	hasDOTnames_list_alias as hasDOTnames_list_alias,
	hasDOTnames_list_str as hasDOTnames_list_str,
	hasDOTnames as hasDOTnames,
	hasDOTop_boolop as hasDOTop_boolop,
	hasDOTop_operator as hasDOTop_operator,
	hasDOTop_unaryop as hasDOTop_unaryop,
	hasDOTop as hasDOTop,
	hasDOToperand as hasDOToperand,
	hasDOTops as hasDOTops,
	hasDOToptional_vars as hasDOToptional_vars,
	hasDOTorelse_expr as hasDOTorelse_expr,
	hasDOTorelse_list_stmt as hasDOTorelse_list_stmt,
	hasDOTorelse as hasDOTorelse,
	hasDOTpattern_pattern as hasDOTpattern_pattern,
	hasDOTpattern_patternORNone as hasDOTpattern_patternORNone,
	hasDOTpattern as hasDOTpattern,
	hasDOTpatterns as hasDOTpatterns,
	hasDOTposonlyargs as hasDOTposonlyargs,
	hasDOTrest as hasDOTrest,
	hasDOTreturns_expr as hasDOTreturns_expr,
	hasDOTreturns_exprORNone as hasDOTreturns_exprORNone,
	hasDOTreturns as hasDOTreturns,
	hasDOTright as hasDOTright,
	hasDOTsimple as hasDOTsimple,
	hasDOTslice as hasDOTslice,
	hasDOTstep as hasDOTstep,
	hasDOTsubject as hasDOTsubject,
	hasDOTtag as hasDOTtag,
	hasDOTtarget as hasDOTtarget,
	hasDOTtargets as hasDOTtargets,
	hasDOTtest as hasDOTtest,
	hasDOTtype as hasDOTtype,
	hasDOTtype_comment as hasDOTtype_comment,
	hasDOTtype_ignores as hasDOTtype_ignores,
	hasDOTtype_params as hasDOTtype_params,
	hasDOTupper as hasDOTupper,
	hasDOTvalue_Any as hasDOTvalue_Any,
	hasDOTvalue_boolORNone as hasDOTvalue_boolORNone,
	hasDOTvalue_expr as hasDOTvalue_expr,
	hasDOTvalue_exprORNone as hasDOTvalue_exprORNone,
	hasDOTvalue as hasDOTvalue,
	hasDOTvalues as hasDOTvalues,
	hasDOTvararg as hasDOTvararg,
)

from mapFolding.someAssemblyRequired._theTypes import (
	ast_expr_Slice as ast_expr_Slice,
	ast_Identifier as ast_Identifier,
	hasDOTtarget_expr as hasDOTtarget_expr,
	hasDOTtarget_Name as hasDOTtarget_Name,
	hasDOTtarget_AttributeORNameORSubscript as hasDOTtarget_AttributeORNameORSubscript,
	ImaCallToName as ImaCallToName,
	intORlist_ast_type_paramORstr_orNone as intORlist_ast_type_paramORstr_orNone,
	intORstr_orNone as intORstr_orNone,
	list_ast_type_paramORstr_orNone as list_ast_type_paramORstr_orNone,
	NodeORattribute as NodeORattribute,
	str_nameDOTname as str_nameDOTname,
	个 as 个,
	)

from mapFolding.someAssemblyRequired._toolboxPython import (
	importLogicalPath2Callable as importLogicalPath2Callable,
	importPathFilename2Callable as importPathFilename2Callable,
	NodeChanger as NodeChanger,
	NodeTourist as NodeTourist,
	parseLogicalPath2astModule as parseLogicalPath2astModule,
	parsePathFilename2astModule as parsePathFilename2astModule,
	)

from mapFolding.someAssemblyRequired._toolboxAntecedents import be as be, DOT as DOT, ifThis as ifThis
from mapFolding.someAssemblyRequired._tool_Make import Make as Make
from mapFolding.someAssemblyRequired._tool_Then import grab as grab, Then as Then

from mapFolding.someAssemblyRequired._toolboxContainers import (
	DeReConstructField2ast as DeReConstructField2ast,
	IngredientsFunction as IngredientsFunction,
	IngredientsModule as IngredientsModule,
	LedgerOfImports as LedgerOfImports,
	RecipeSynthesizeFlow as RecipeSynthesizeFlow,
	ShatteredDataclass as ShatteredDataclass,
)

from mapFolding.someAssemblyRequired._toolboxAST import (
	astModuleToIngredientsFunction as astModuleToIngredientsFunction,
	extractClassDef as extractClassDef,
	extractFunctionDef as extractFunctionDef,
	)
