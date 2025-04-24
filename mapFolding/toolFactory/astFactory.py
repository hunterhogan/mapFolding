"""
This module programmatically generates Python source files that provide type-guard and constructor utilities for AST
node types, based on a given AST stub file. It is designed to automate the creation of two classes: `be` (type-guards
for AST node types) and `Make` (constructors for AST node types), each with a large number of static methods
corresponding to AST classes. The output files are written to a location determined by configuration and logical path
infix.

The intended audience is future me and future AI assistants: this module is a critical automation point for extending or
regenerating the type-guard and constructor utilities. If you are adding new AST node types, changing the structure of
the AST, or updating the logic for filtering or method generation, this is the place to do it. The code is organized for
clarity, extensibility, and minimal manual intervention. All identifier and code style conventions are chosen for
maximum semantic clarity and future maintainability.
"""
from mapFolding import The, writeStringToHere
from pathlib import PurePosixPath
from typing import cast, TypeAlias as typing_TypeAlias
import ast

ast_Identifier: typing_TypeAlias = str
str_nameDOTname: typing_TypeAlias = str
ast_expr_Slice: typing_TypeAlias = ast.expr

sys_version_infoTarget: tuple[int, int] = (3, 13)

list_astDOTStuPydHARDCODED: list[ast_Identifier] = ['astDOTParamSpec', 'astDOTTryStar', 'astDOTTypeAlias', 'astDOTTypeVar', 'astDOTTypeVarTuple', 'astDOTtype_param']
list_astDOTStuPyd = list_astDOTStuPydHARDCODED

class SubstituteTyping_TypeAlias(ast.NodeTransformer):
	"""
	Transformer that substitutes type alias placeholders in AST Name nodes.

	This class extends ast.NodeTransformer and overrides the visit_Name method to
	replace specific placeholder identifiers:
		- '_Identifier' → 'ast_Identifier'
		- '_Slice'      → 'ast_expr_Slice'

	By applying this transformer to an AST, any Name nodes using these
	placeholders will be updated to their corresponding AST-typed alias names.
	"""
	def visit_Name(self, node: ast.Name) -> ast.Name:
		if node.id == '_Identifier':
			node.id = 'ast_Identifier'
		elif node.id == '_Slice':
			node.id = 'ast_expr_Slice'
		return node

class MakeDictionaryOfAnnotations(ast.NodeVisitor):
	"""
	MakeDictionaryOfAnnotations(ast.NodeVisitor)

	A visitor that traverses an AST and builds a dictionary mapping annotation names
	to their corresponding `ast.Attribute` nodes under the `ast` module.
	Attributes:
		astAST (ast.AST):
			The root AST node to be visited.
		dictionarySubstitutions (dict[str, ast.Attribute]):
			A mapping where each key is an annotation identifier (e.g. class name
			or `_Pattern`) and each value is an `ast.Attribute` node pointing to
			`ast.<Identifier>` (e.g. `ast.ClassDef`, `ast.pattern`).

	Methods:
		visit_ClassDef(node: ast.ClassDef) -> None:
			On encountering a class definition in the AST, inserts an entry in
			`dictionarySubstitutions` mapping the class name to
			`ast.<ClassName>`.
		getDictionary() -> dict[str, ast.Attribute]:
			Triggers a traversal of `astAST` (via `self.visit`) and returns the
			completed `dictionarySubstitutions` mapping.
	"""
	def __init__(self, astAST: ast.AST) -> None:
		super().__init__()
		self.astAST = astAST
		self.dictionarySubstitutions: dict[ast_Identifier, ast.Attribute] = {'_Pattern': ast.Attribute(value=ast.Name(id='ast'), attr='pattern', ctx=ast.Load())}

	def visit_ClassDef(self, node: ast.ClassDef) -> None:
		self.dictionarySubstitutions[node.name] = ast.Attribute(value=ast.Name(id='ast'), attr=node.name, ctx=ast.Load())

	def getDictionary(self) -> dict[str, ast.Attribute]:
		self.visit(self.astAST)
		return self.dictionarySubstitutions

class ChangeName2Attribute(ast.NodeTransformer):
	"""
	Transform AST Name nodes into AST Attribute nodes based on a substitution map.

	This NodeTransformer scans the AST for ast.Name nodes and, whenever a name
	matches a key in the provided substitution dictionary, replaces it with the
	corresponding ast.Attribute node.

	Parameters:
		dictionarySubstitutions (dict[ast_Identifier, ast.Attribute]):
			A mapping from identifier names (or ast_Identifier objects) to their
			replacement ast.Attribute nodes.

	Methods:
		visit_Name(node: ast.Name) -> ast.Attribute | ast.Name:
			If node.id is found in dictionarySubstitutions, returns the mapped
			ast.Attribute. Otherwise, returns the original ast.Name node.
	"""
	def __init__(self, dictionarySubstitutions: dict[ast_Identifier, ast.Attribute]) -> None:
		super().__init__()
		self.dictionarySubstitutions = dictionarySubstitutions

	def visit_Name(self, node: ast.Name) -> ast.Attribute | ast.Name:
		if node.id in self.dictionarySubstitutions:
			return self.dictionarySubstitutions[node.id]
		return node

def makeTools(astStubFile: ast.AST, logicalPathInfix: str_nameDOTname) -> None:
	"""
	How does this function work?

	Filtering.
	"""
	def writeModule(astModule: ast.Module, moduleIdentifier: ast_Identifier) -> None:
		"""
		Generate and write a Python module from its AST representation.

		This function performs the following steps:
		1. Ensures all AST nodes have location information by calling ast.fix_missing_locations().
		2. Converts the AST back into source code using ast.unparse().
		3. Constructs an output file path by combining a fixed prefix, the provided module identifier,
			a logical infix path, and the configured package path and file extension.
		4. Writes the generated source code to the target file using writeStringToHere().

		Args:
			astModule (ast.Module): The abstract syntax tree of the module to be written.
			moduleIdentifier (ast_Identifier): A unique identifier used to name the output file.

		Returns:
			 None:
		"""
		ast.fix_missing_locations(astModule)
		pythonSource: str = ast.unparse(astModule)
		moduleIdentifierPrefix: str = '_tool_'
		pathFilenameModule = PurePosixPath(The.pathPackage, logicalPathInfix, moduleIdentifierPrefix + moduleIdentifier + The.fileExtension)
		writeStringToHere(pythonSource, pathFilenameModule)

	beClassDef = ast.ClassDef(name='be', bases=[], keywords=[], body=[], decorator_list=[])
	# This function creates 126 `staticmethod` in the `be` class.
	MakeClassDef = ast.ClassDef(name='Make', bases=[], keywords=[], body=[], decorator_list=[])
	# This function creates 76 `staticmethod` in the `Make` class.
	# How can this module create more than 200 methods in two classes? Filtering.

	dictionaryAnnotations: dict[ast_Identifier, ast.Attribute] = MakeDictionaryOfAnnotations(astStubFile).getDictionary()

	# Work on one ClassDef at a time.
	for node in ast.walk(astStubFile):
		# Filter out undesired nodes.
		if not isinstance(node, ast.ClassDef):
			continue
		if any(isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == 'deprecated' for decorator in node.decorator_list):
			continue
		if node.name.startswith('_'):
			continue

		# Change the identifier solely for the benefit of clarity as you read this code.
		ImaClassDef = node

		# Create ast "fragments" before you need them.
		ClassDefIdentifier: ast_Identifier = ImaClassDef.name
		class_nameDOTname: str_nameDOTname = 'ast.' + ClassDefIdentifier
		for pyDOTwhy in list_astDOTStuPyd:
			# This ties into a system to allow the package to run on Python < 3.13.
			astClass = pyDOTwhy.replace('DOT', '.')
			class_nameDOTname = class_nameDOTname.replace(astClass, pyDOTwhy)
		ClassDefName = ast.Name(class_nameDOTname)

		# Create the ClassDef and add directly to the body of the class.
		beClassDef.body.append(ast.FunctionDef(name=ClassDefIdentifier, args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='node', annotation=ast.Name('ast.AST'))], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[ast.Return(value=ast.Call(func=ast.Name('isinstance'), args=[ast.Name('node'), ClassDefName], keywords=[]))], decorator_list=[ast.Name('staticmethod')], type_comment=None, returns=ast.Subscript(value=ast.Name('TypeGuard'), slice=ClassDefName, ctx=ast.Load())))

		# Start: cope with different arguments for Python versions. ==============================================================
		# NOTE: I would love suggestions to improve this section.
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

		# End: cope with different arguments for Python versions. ============================================================

		# Strongly prefer to use ast to make ast: avoid intermediate primitive types.
		MakeClassDef.body.append(ast.FunctionDef(name=ClassDefIdentifier
			, args=ast.arguments(posonlyargs=[]
				, args=[]
				, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
			, body=[ast.Return(value=ast.Call(func=ClassDefName
				, args=[]
				, keywords=[]))], decorator_list=[ast.Name('staticmethod')], type_comment=None, returns=ClassDefName))

		SubstituteTyping_TypeAlias().visit(ImaClassDef)
		for attribute in listAttributes:
			for subnode in ast.walk(ImaClassDef):
				if isinstance(subnode, ast.AnnAssign) and isinstance(subnode.target, ast.Name) and subnode.target.id == attribute:
					annotation = ChangeName2Attribute(dictionaryAnnotations).visit(subnode.annotation)
					cast(ast.FunctionDef, MakeClassDef.body[-1]).args.args.append(ast.arg(arg=attribute, annotation=annotation))
					cast(ast.Call, cast(ast.Return, cast(ast.FunctionDef, MakeClassDef.body[-1]).body[0]).value).args.append(ast.Name(attribute, ctx=ast.Load()))

	beClassDef.body.sort(key=lambda astFunctionDef: astFunctionDef.name.lower()) # pyright: ignore[reportAttributeAccessIssue, reportUnknownLambdaType, reportUnknownMemberType]
	MakeClassDef.body.sort(key=lambda astFunctionDef: astFunctionDef.name.lower()) # pyright: ignore[reportAttributeAccessIssue, reportUnknownLambdaType, reportUnknownMemberType]

	beClassDef.body.insert(0, ast.Expr(value=ast.Constant(value=beClassDocstring)))
	MakeClassDef.body.insert(0, ast.Expr(value=ast.Constant(value=MakeClassDocstring)))

	be_astModule = ast.Module(
		body=[ast.Expr(ast.Constant(docstringWarning))
			, ast.ImportFrom('mapFolding', [ast.alias(pleasedonotcrashwhileimportingtypes) for pleasedonotcrashwhileimportingtypes in list_astDOTStuPyd], 0)
			, ast.ImportFrom('typing', [ast.alias('TypeGuard')], 0)
			, ast.Import([ast.alias('ast')])
			, beClassDef
			],
		type_ignores=[]
	)
	writeModule(be_astModule, beClassDef.name)

	Make_astModule = ast.Module(
		body=[ast.Expr(ast.Constant(docstringWarning))
			, ast.ImportFrom('mapFolding', [ast.alias(pleasedonotcrashwhileimportingtypes) for pleasedonotcrashwhileimportingtypes in list_astDOTStuPyd], 0)
			, ast.ImportFrom('mapFolding.someAssemblyRequired', [ast.alias('ast_Identifier'), ast.alias('ast_expr_Slice')], 0)
			, ast.ImportFrom('typing', [ast.alias('Any'), ast.alias('Literal')], 0)
			, ast.Import([ast.alias('ast')])
			, MakeClassDef
			],
		type_ignores=[]
	)
	writeModule(Make_astModule, MakeClassDef.name)

docstringWarning: str = """This file is generated automatically, so changes to this file will be lost."""

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
