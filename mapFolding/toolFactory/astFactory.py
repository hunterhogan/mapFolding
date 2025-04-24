from mapFolding import The, writeStringToHere
from pathlib import PurePosixPath
from typing import TypeAlias as typing_TypeAlias
import ast

ast_Identifier: typing_TypeAlias = str
str_nameDOTname: typing_TypeAlias = str
ast_expr_Slice: typing_TypeAlias = ast.expr

class substituteTyping_TypeAlias(ast.NodeTransformer):
	def visit_Name(self, node: ast.Name) -> ast.Name:
		if node.id == '_Identifier':
			node.id = 'ast_Identifier'
		elif node.id == '_Slice':
			node.id = 'ast_expr_Slice'
		return node

class Z0Z_Name2Attribute(ast.NodeTransformer):
	def __init__(self, dictionarySubstitutions: dict[ast_Identifier, ast.Attribute]) -> None:
		super().__init__()
		self.dictionarySubstitutions = dictionarySubstitutions

	def visit_Name(self, node: ast.Name) -> ast.Attribute | ast.Name:
		if node.id in self.dictionarySubstitutions:
			return self.dictionarySubstitutions[node.id]
		return node

class Z0Z_dictionary(ast.NodeVisitor):
    def __init__(self, astAST: ast.AST):
        super().__init__()
        self.astAST = astAST
        self.dictionarySubstitutions: dict[ast_Identifier, ast.Attribute] = {'_Pattern': ast.Attribute(value=ast.Name(id='ast'), attr='pattern', ctx=ast.Load())}

    def visit_ClassDef(self, node: ast.ClassDef):
        self.dictionarySubstitutions[node.name] = ast.Attribute(value=ast.Name(id='ast'), attr=node.name, ctx=ast.Load())

    def get_dictionarySubstitutions(self):
        self.visit(self.astAST)
        return self.dictionarySubstitutions

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

	dictionaryAnnotations: dict[ast_Identifier, ast.Attribute] = Z0Z_dictionary(astStubFile).get_dictionarySubstitutions()
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

		substituteTyping_TypeAlias().visit(ImaClassDef)
		# Z0Z_Name2Attribute(dictionaryAnnotations).visit(ImaClassDef)
		list_ast_arg: list[ast.arg] = []
		listName4Call: list[ast.expr] = []
		for attribute in listAttributes:
			for subnode in ast.walk(ImaClassDef):
				if isinstance(subnode, ast.AnnAssign) and isinstance(subnode.target, ast.Name) and subnode.target.id == attribute:
					annotation = Z0Z_Name2Attribute(dictionaryAnnotations).visit(subnode.annotation)
					list_ast_arg.append(ast.arg(arg=attribute, annotation=annotation))
					listName4Call.append(ast.Name(attribute, ctx=ast.Load()))

		MakeClassDef.body.append(ast.FunctionDef(name=ClassDefIdentifier, args=ast.arguments(posonlyargs=[], args=list_ast_arg, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
			, body=[ast.Return(value=ast.Call(func=ClassDefName
										, args=listName4Call
										, keywords=[]))]
			, decorator_list=[ast.Name('staticmethod')], type_comment=None
			, returns=ClassDefName))

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
		body=[ast.Expr(ast.Constant(docstringWarning.strip()))
			, ast.ImportFrom('mapFolding', [ast.alias(pleasedonotcrashwhileimportingtypes) for pleasedonotcrashwhileimportingtypes in list_astDOTStuPyd], 0)
			, ast.Import([ast.alias('ast')])
			, ast.ImportFrom('typing', [ast.alias('Any'), ast.alias('Literal')], 0)
			, ast.ImportFrom('mapFolding.someAssemblyRequired', [ast.alias('ast_Identifier'), ast.alias('ast_expr_Slice')], 0)
			, MakeClassDef
			],
		type_ignores=[]
	)

	writeModule(Make_astModule, MakeClassDef.name)
