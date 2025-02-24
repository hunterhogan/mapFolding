from collections.abc import Sequence
from mapFolding import hackSSOTdatatype, ParametersNumba, parametersNumbaDEFAULT, Z0Z_getDatatypeModuleScalar, Z0Z_getDecoratorCallable
from mapFolding.someAssemblyRequired.synthesizeGeneric import *
from mapFolding.someAssemblyRequired.synthesizeGeneric import UniversalImportTracker
from typing import cast
import ast

def thisIsNumbaDotJit(Ima: ast.AST) -> bool:
	return ifThis.isCallWithAttribute(Z0Z_getDatatypeModuleScalar(), Z0Z_getDecoratorCallable())(Ima)

def thisIsJit(Ima: ast.AST) -> bool:
	return ifThis.isCallWithName(Z0Z_getDecoratorCallable())(Ima)

def thisIsAnyNumbaJitDecorator(Ima: ast.AST) -> bool:
	return thisIsNumbaDotJit(Ima) or thisIsJit(Ima)

def decorateCallableWithNumba(FunctionDefTarget: ast.FunctionDef, allImports: UniversalImportTracker, parametersNumba: ParametersNumba | None = None) -> tuple[ast.FunctionDef, UniversalImportTracker]:
	def Z0Z_UnhandledDecorators(astCallable: ast.FunctionDef) -> ast.FunctionDef:
		# TODO: more explicit handling of decorators. I'm able to ignore this because I know `algorithmSource` doesn't have any decorators.
		for decoratorItem in astCallable.decorator_list.copy():
			import warnings
			astCallable.decorator_list.remove(decoratorItem)
			warnings.warn(f"Removed decorator {ast.unparse(decoratorItem)} from {astCallable.name}")
		return astCallable

	def make_arg4parameter(signatureElement: ast.arg):
		if isinstance(signatureElement.annotation, ast.Subscript) and isinstance(signatureElement.annotation.slice, ast.Tuple):
			annotationShape = signatureElement.annotation.slice.elts[0]
			if isinstance(annotationShape, ast.Subscript) and isinstance(annotationShape.slice, ast.Tuple):
				shapeAsListSlices = [ast.Slice() for _axis in range(len(annotationShape.slice.elts))]
				shapeAsListSlices[-1] = ast.Slice(step=ast.Constant(value=1))
				shapeAST = ast.Tuple(elts=list(shapeAsListSlices), ctx=ast.Load())
			else:
				shapeAST = ast.Slice(step=ast.Constant(value=1))

			annotationDtype = signatureElement.annotation.slice.elts[1]
			if (isinstance(annotationDtype, ast.Subscript) and isinstance(annotationDtype.slice, ast.Attribute)):
				datatypeAST = annotationDtype.slice.attr
			else:
				datatypeAST = None

			ndarrayName = signatureElement.arg
			Z0Z_hacky_dtype = hackSSOTdatatype(ndarrayName)
			datatype_attr = datatypeAST or Z0Z_hacky_dtype
			allImports.addImportFromStr(datatypeModuleDecorator, datatype_attr)
			datatypeNumba = ast.Name(id=datatype_attr, ctx=ast.Load())

			return ast.Subscript(value=datatypeNumba, slice=shapeAST, ctx=ast.Load())

		elif isinstance(signatureElement.annotation, ast.Name):
			return signatureElement.annotation
		return

	datatypeModuleDecorator: str = Z0Z_getDatatypeModuleScalar()
	list_argsDecorator: Sequence[ast.expr] = []

	list_arg4signature_or_function: list[ast.expr] = []
	for parameter in FunctionDefTarget.args.args:
		signatureElement = make_arg4parameter(parameter)
		if signatureElement:
			list_arg4signature_or_function.append(signatureElement)

	if FunctionDefTarget.returns and isinstance(FunctionDefTarget.returns, ast.Name):
		theReturn: ast.Name = FunctionDefTarget.returns
		list_argsDecorator = [cast(ast.expr, ast.Call(func=ast.Name(id=theReturn.id, ctx=ast.Load())
							, args=list_arg4signature_or_function if list_arg4signature_or_function else [], keywords=[] ) )]
	elif list_arg4signature_or_function:
		list_argsDecorator = [cast(ast.expr, ast.Tuple(elts=list_arg4signature_or_function, ctx=ast.Load()))]

	for decorator in FunctionDefTarget.decorator_list.copy():
		if thisIsAnyNumbaJitDecorator(decorator):
			decorator = cast(ast.Call, decorator)
			if parametersNumba is None:
				parametersNumbaSherpa = Then.copy_astCallKeywords(decorator)
				if (_HunterIsSureThereAreBetterWaysToDoThis := True):
					if parametersNumbaSherpa:
						parametersNumba = cast(ParametersNumba, parametersNumbaSherpa)
		FunctionDefTarget.decorator_list.remove(decorator)

	FunctionDefTarget = Z0Z_UnhandledDecorators(FunctionDefTarget)
	if parametersNumba is None:
		parametersNumba = parametersNumbaDEFAULT
	listDecoratorKeywords = [ast.keyword(arg=parameterName, value=ast.Constant(value=parameterValue)) for parameterName, parameterValue in parametersNumba.items()]

	decoratorModule = Z0Z_getDatatypeModuleScalar()
	decoratorCallable = Z0Z_getDecoratorCallable()
	allImports.addImportFromStr(decoratorModule, decoratorCallable)
	astDecorator = Then.make_astCall(decoratorCallable, list_argsDecorator, listDecoratorKeywords, None)

	FunctionDefTarget.decorator_list = [astDecorator]
	return FunctionDefTarget, allImports
