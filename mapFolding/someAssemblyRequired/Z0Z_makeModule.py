from mapFolding.someAssemblyRequired import LedgerOfImports, ast_Identifier, IngredientsFunction, shatter_dataclassesDOTdataclass
from mapFolding import ComputationState, Z0Z_dispatcherOfDataCallable, Z0Z_dispatcherOfDataFilename, FREAKOUT
import inspect
import ast

# pyright: standard

def Z0Z_makeDataDispatcher():
	dataclassesDOTdataclass = ComputationState

	ingredientsDispatcherOfData = IngredientsFunction(ast.FunctionDef(Z0Z_dispatcherOfDataCallable, ast.arguments([], [], None, [], [], None, []), [], []))

	astClassDef = ast.parse(inspect.getsource(dataclassesDOTdataclass)).body[0]
	if not isinstance(astClassDef, ast.ClassDef): raise FREAKOUT

	instance_Identifier = 'state'
	listAnnAssign = shatter_dataclassesDOTdataclass(astClassDef, instance_Identifier)
