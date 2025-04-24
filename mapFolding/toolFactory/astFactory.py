from mapFolding import The, writeStringToHere
from pathlib import PurePosixPath
from typing import TypeAlias as typing_TypeAlias
import ast

ast_Identifier: typing_TypeAlias = str
str_nameDOTname: typing_TypeAlias = str

sys_version_infoTarget: tuple[int, int] = (3, 13)

docstringWarning: str = """This file is generated automatically, so changes to this file will be lost."""

list_astDOTStuPydHARDCODED: list[ast_Identifier] = ['astDOTParamSpec', 'astDOTTryStar', 'astDOTTypeAlias', 'astDOTTypeVar', 'astDOTTypeVarTuple', 'astDOTtype_param']
list_astDOTStuPyd = list_astDOTStuPydHARDCODED

def makeTools(astStubFile: ast.AST, logicalPathInfix: str_nameDOTname):
	def writeModule(astModule: ast.Module, moduleIdentifier: ast_Identifier):
		ast.fix_missing_locations(astModule)
		pythonSource: str = ast.unparse(astModule)
		moduleIdentifierPrefix: str = '_tool_'
		pathFilenameModule = PurePosixPath(The.pathPackage, logicalPathInfix, moduleIdentifierPrefix + moduleIdentifier + The.fileExtension)
		writeStringToHere(pythonSource, pathFilenameModule)

	beClassDef = ast.ClassDef(name='be', bases=[], keywords=[], body=[], decorator_list=[])
	MakeClassDef = ast.ClassDef(name='Make', bases=[], keywords=[], body=[], decorator_list=[])
	"""
	Notes for the logic to create the `Make` class:
		- Once you have the list of attributes, you need to get the type(s) for each attribute.
		- All types are in `ast.AnnAssign`. The most basic format is:
		```
		AnnAssign(
			target=Name(id='name', ctx=Store()),
			annotation=Name(id='_Identifier', ctx=Load()),
			simple=1)
		```
		- The target is always `ast.Name` and the id matches the attribute name.
		- The annotation can get complicated. In a perfect world, we could just use the annotation from the
		`ast.AnnAssign` as the annotation in the method. But, their annotations say "stmt", for example, not "ast.stmt",
		so we will need to make some adjustments.
		- And there is another complication: some `ast.AnnAssign` are inside a system version check, such as:
		```
		If(
			test=Compare(
				left=Attribute(
					value=Name(id='sys', ctx=Load()),
					attr='version_info',
					ctx=Load()),
				ops=[
					GtE()],
				comparators=[
					Tuple(
						elts=[
							Constant(value=3),
							Constant(value=12)],
						ctx=Load())]),
			body=[
				AnnAssign(
					target=Name(id='type_params', ctx=Store()),
					annotation=Subscript(
						value=Name(id='list', ctx=Load()),
						slice=Name(id='type_param', ctx=Load()),
						ctx=Load()),
					simple=1)]),
		```

	You MUST use ast to do the work. Do not use string manipulation. The ast module is your friend.

	"""

	for node in ast.walk(astStubFile):
		if not isinstance(node, ast.ClassDef):
			continue
		if any(isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == 'deprecated' for decorator in node.decorator_list):
			continue
		if node.name.startswith('_'):
			continue
		ImaClassDef = node
		ClassDefIdentifier: ast_Identifier = ImaClassDef.name
		class_nameDOTname: str_nameDOTname = 'ast.' + ClassDefIdentifier
		for pyDOTwhy in list_astDOTStuPyd:
			astClass = pyDOTwhy.replace('DOT', '.')
			class_nameDOTname = class_nameDOTname.replace(astClass, pyDOTwhy)
		ClassDefName = ast.Name(class_nameDOTname)

		beClassDef.body.append(ast.FunctionDef(name=ClassDefIdentifier, args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='node', annotation=ast.Name('ast.AST'))], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[ast.Return(value=ast.Call(func=ast.Name('isinstance'), args=[ast.Name('node'), ClassDefName], keywords=[]))], decorator_list=[ast.Name('staticmethod')], type_comment=None, returns=ast.Subscript(value=ast.Name('TypeGuard'), slice=ClassDefName, ctx=ast.Load())))

		listAttributes: list[ast_Identifier] = []
		list__match_args__: list[list[ast_Identifier]] = []
		dictAttributes: dict[tuple[int, int], list[ast_Identifier]] = {}
		for subnode in ast.walk(ImaClassDef):
			listAttributes = []
			if (isinstance(subnode, ast.If) and isinstance(subnode.test, ast.Compare)
				and isinstance(subnode.test.left, ast.Attribute)
				and subnode.test.left.attr == 'version_info' and isinstance(subnode.test.comparators[0], ast.Tuple)
				and isinstance(subnode.body[0], ast.Assign) and isinstance(subnode.body[0].targets[0], ast.Name) and subnode.body[0].targets[0].id == '__match_args__'
				and isinstance(subnode.body[0].value, ast.Tuple) and subnode.body[0].value.elts):
				sys_version_info: tuple[int, int] = ast.literal_eval(subnode.test.comparators[0])
				if sys_version_info > sys_version_infoTarget:
					continue
				if any(sys_version_info < key for key in dictAttributes.keys()): # pyright: ignore[reportOperatorIssue]
					continue
				dictAttributes[sys_version_info] = []
				for astAST in subnode.body[0].value.elts:
					if isinstance(astAST, ast.Constant):
						dictAttributes[sys_version_info].append(astAST.value)
				if sys_version_info == sys_version_infoTarget:
					break

			if (isinstance(subnode, ast.Assign) and isinstance(subnode.targets[0], ast.Name) and subnode.targets[0].id == '__match_args__'
				and isinstance(subnode.value, ast.Tuple) and subnode.value.elts):
				for astAST in subnode.value.elts:
					if isinstance(astAST, ast.Constant):
						listAttributes.append(astAST.value)
				list__match_args__.append(listAttributes)

		if not list__match_args__ and not dictAttributes and not listAttributes:
			continue
		elif sys_version_infoTarget in dictAttributes:
			listAttributes = dictAttributes[sys_version_infoTarget]
		elif dictAttributes:
			listAttributes = dictAttributes[max(dictAttributes.keys())]
		elif len(list__match_args__) == 1:
			listAttributes = list__match_args__[0]
		else:
			raise Exception(f"Hunter did not predict this situation.\n\t{ClassDefIdentifier = }\n\t{list__match_args__ = }\n\t{dictAttributes = }")

	beClassDef.body.sort(key=lambda astFunctionDef: astFunctionDef.name.lower()) # pyright: ignore[reportAttributeAccessIssue, reportUnknownLambdaType, reportUnknownMemberType]
	MakeClassDef.body.sort(key=lambda astFunctionDef: astFunctionDef.name.lower()) # pyright: ignore[reportAttributeAccessIssue, reportUnknownLambdaType, reportUnknownMemberType]

	beClassDocstring: str = (
	"""
	Provide type-guard functions for safely verifying AST node types during manipulation.

	The be class contains static methods that perform runtime type verification of AST nodes, returning TypeGuard
	results that enable static type checkers to narrow node types in conditional branches. These type-guards:

	1. Improve code safety by preventing operations on incompatible node types.
	2. Enable IDE tooling to provide better autocompletion and error detection.
	3. Document expected node types in a way that is enforced by the type system.
	4. Support pattern-matching workflows where node types must be verified before access.

	When used with conditional statements, these type-guards allow for precise, type-safe manipulation of AST nodes
	while maintaining full static type checking capabilities, even in complex transformation scenarios.
	"""
	)
	beClassDef.body.insert(0, ast.Expr(value=ast.Constant(value=beClassDocstring)))

	MakeClassDocstring: str = (
	"""
	Almost all parameters described here are only accessible through a method's `**keywordArguments` parameter.

	Parameters:
		context (ast.Load()): Are you loading from, storing to, or deleting the identifier? The `context` (also, `ctx`) value is `ast.Load()`, `ast.Store()`, or `ast.Del()`.
		col_offset (0): int Position information specifying the column where an AST node begins.
		end_col_offset (None): int|None Position information specifying the column where an AST node ends.
		end_lineno (None): int|None Position information specifying the line number where an AST node ends.
		level (0): int Module import depth level that controls relative vs absolute imports. Default 0 indicates absolute import.
		lineno: int Position information manually specifying the line number where an AST node begins.
		kind (None): str|None Used for type annotations in limited cases.
		type_comment (None): str|None "type_comment is an optional string with the type annotation as a comment." or `# type: ignore`.
		type_params: list[ast.type_param] Type parameters for generic type definitions.

	The `ast._Attributes`, lineno, col_offset, end_lineno, and end_col_offset, hold position information; however, they are, importantly, _not_ `ast._fields`.
	"""
	)

	MakeClassDef.body.insert(0, ast.Expr(value=ast.Constant(value=MakeClassDocstring)))

	be_astModule = ast.Module(
		body=[ast.Expr(ast.Constant(docstringWarning.strip())), ast.ImportFrom('mapFolding', [ast.alias(pleasedonotcrashwhileimportingtypes) for pleasedonotcrashwhileimportingtypes in list_astDOTStuPyd], 0), ast.Import([ast.alias('ast')])
			, ast.ImportFrom('typing', [ast.alias('TypeGuard')], 0)
			, beClassDef
			],
		type_ignores=[]
	)

	writeModule(be_astModule, beClassDef.name)

	Make_astModule = ast.Module(
		body=[ast.Expr(ast.Constant(docstringWarning.strip())), ast.ImportFrom('mapFolding', [ast.alias(pleasedonotcrashwhileimportingtypes) for pleasedonotcrashwhileimportingtypes in list_astDOTStuPyd], 0), ast.Import([ast.alias('ast')])
			, ast.ImportFrom('typing', [ast.alias('Any')], 0)
			, ast.ImportFrom('mapFolding.someAssemblyRequired', [ast.alias('ast_Identifier')], 0)
			, MakeClassDef
			],
		type_ignores=[]
	)

	writeModule(Make_astModule, MakeClassDef.name)
