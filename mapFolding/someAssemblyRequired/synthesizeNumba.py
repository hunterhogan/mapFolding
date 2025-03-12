from collections.abc import Sequence
from mapFolding.someAssemblyRequired.whatWillBe import LedgerOfImports, ParametersNumba, Z0Z_getDatatypeModuleScalar, parametersNumbaDEFAULT
from mapFolding.someAssemblyRequired.whatWillBe import Z0Z_getDecoratorCallable
from mapFolding.someAssemblyRequired import Make, ifThis
from typing import Any, cast
import ast

def thisIsNumbaDotJit(Ima: ast.AST) -> bool:
	return ifThis.isCallNamespace_Identifier(Z0Z_getDatatypeModuleScalar(), Z0Z_getDecoratorCallable())(Ima)

def thisIsJit(Ima: ast.AST) -> bool:
	return ifThis.isCall_Identifier(Z0Z_getDecoratorCallable())(Ima)

def thisIsAnyNumbaJitDecorator(Ima: ast.AST) -> bool:
	return thisIsNumbaDotJit(Ima) or thisIsJit(Ima)

def decorateCallableWithNumba(FunctionDefTarget: ast.FunctionDef, allImports: LedgerOfImports, parametersNumba: ParametersNumba | None = None) -> tuple[ast.FunctionDef, LedgerOfImports]:
	def Z0Z_UnhandledDecorators(astCallable: ast.FunctionDef) -> ast.FunctionDef:
		# TODO: more explicit handling of decorators. I'm able to ignore this because I know `algorithmSource` doesn't have any decorators.
		for decoratorItem in astCallable.decorator_list.copy():
			import warnings
			astCallable.decorator_list.remove(decoratorItem)
			warnings.warn(f"Removed decorator {ast.unparse(decoratorItem)} from {astCallable.name}")
		return astCallable

	def make_arg4parameter(signatureElement: ast.arg) -> ast.Subscript | ast.Name | None:
		if isinstance(signatureElement.annotation, ast.Subscript) and isinstance(signatureElement.annotation.slice, ast.Tuple):
			annotationShape: ast.expr = signatureElement.annotation.slice.elts[0]
			if isinstance(annotationShape, ast.Subscript) and isinstance(annotationShape.slice, ast.Tuple):
				shapeAsListSlices: list[ast.Slice] = [ast.Slice() for _axis in range(len(annotationShape.slice.elts))]
				shapeAsListSlices[-1] = ast.Slice(step=ast.Constant(value=1))
				shapeAST: ast.Slice | ast.Tuple = ast.Tuple(elts=list(shapeAsListSlices), ctx=ast.Load())
			else:
				shapeAST = ast.Slice(step=ast.Constant(value=1))

			annotationDtype: ast.expr = signatureElement.annotation.slice.elts[1]
			if (isinstance(annotationDtype, ast.Subscript) and isinstance(annotationDtype.slice, ast.Attribute)):
				datatypeAST = annotationDtype.slice.attr
			else:
				datatypeAST = None

			ndarrayName = signatureElement.arg
			Z0Z_hacky_dtype: str = ndarrayName
			datatype_attr = datatypeAST or Z0Z_hacky_dtype
			allImports.addImportFromStr(datatypeModuleDecorator, datatype_attr)
			datatypeNumba = ast.Name(id=datatype_attr, ctx=ast.Load())

			return ast.Subscript(value=datatypeNumba, slice=shapeAST, ctx=ast.Load())

		elif isinstance(signatureElement.annotation, ast.Name):
			return signatureElement.annotation
		return None

	datatypeModuleDecorator: str = Z0Z_getDatatypeModuleScalar()
	list_argsDecorator: Sequence[ast.expr] = []

	list_arg4signature_or_function: list[ast.expr] = []
	for parameter in FunctionDefTarget.args.args:
		signatureElement: ast.Subscript | ast.Name | None = make_arg4parameter(parameter)
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
				parametersNumbaSherpa: dict[str, Any] = Make.copy_astCallKeywords(decorator)
				if (_HunterIsSureThereAreBetterWaysToDoThis := True):
					if parametersNumbaSherpa:
						parametersNumba = cast(ParametersNumba, parametersNumbaSherpa)
		FunctionDefTarget.decorator_list.remove(decorator)

	FunctionDefTarget = Z0Z_UnhandledDecorators(FunctionDefTarget)
	if parametersNumba is None:
		parametersNumba = parametersNumbaDEFAULT
	listDecoratorKeywords: list[ast.keyword] = [ast.keyword(arg=parameterName, value=ast.Constant(value=parameterValue)) for parameterName, parameterValue in parametersNumba.items()]

	decoratorModule: str = Z0Z_getDatatypeModuleScalar()
	decoratorCallable: str = Z0Z_getDecoratorCallable()
	allImports.addImportFromStr(decoratorModule, decoratorCallable)
	astDecorator: ast.Call = Make.astCall(Make.astName(decoratorCallable), list_argsDecorator, listDecoratorKeywords)

	FunctionDefTarget.decorator_list = [astDecorator]
	return FunctionDefTarget, allImports
