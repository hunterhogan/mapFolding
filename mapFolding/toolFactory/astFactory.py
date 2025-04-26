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
from mapFolding.toolFactory.astFactory_annex import handmadeMethods_grab, handmadeTypeAlias_astTypes, MakeAttributeFunctionDef, MakeImportFunctionDef
from mapFolding.toolFactory.astFactory_docstrings import docstringWarning, beClassDefDocstring, DOTClassDefDocstring, grabClassDefDocstring, MakeClassDefDocstring
from pathlib import PurePosixPath
from string import ascii_letters
from typing import Literal, cast, TypeAlias as typing_TypeAlias
import ast

# TODO this is not DRY, but you can't import from some assembly required
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
		elif node.id == 'str':
			node.id = 'ast_Identifier'
		elif node.id == '_Slice':
			node.id = 'ast_expr_Slice'
		return node

class MakeDictionaryOf_astClassAnnotations(ast.NodeVisitor):
	"""
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
		self.dictionarySubstitutions: dict[ast_Identifier, ast.Name | ast.Attribute] = {'_Pattern': ast.Attribute(value=ast.Name('ast'), attr='pattern', ctx=ast.Load())}

	def visit_ClassDef(self, node: ast.ClassDef) -> None:
		if 'astDOT' + node.name in list_astDOTStuPyd:
			NameOrAttribute = ast.Name('astDOT' + node.name, ctx=ast.Load())
		else:
			NameOrAttribute = ast.Attribute(value=ast.Name('ast'), attr=node.name, ctx=ast.Load())
		self.dictionarySubstitutions[node.name] = NameOrAttribute

	def getDictionary(self) -> dict[ast_Identifier, ast.Name | ast.Attribute]:
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
	def __init__(self, dictionarySubstitutions: dict[ast_Identifier, ast.Name | ast.Attribute]) -> None:
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

		Parameters:
			astModule (ast.Module): The abstract syntax tree of the module to be written.
			moduleIdentifier (ast_Identifier): A unique identifier used to name the output file.

		Returns:
			None:
		"""
		ast.fix_missing_locations(astModule)
		pythonSource: str = ast.unparse(astModule)
		pathFilenameModule = PurePosixPath(The.pathPackage, logicalPathInfix, moduleIdentifier + The.fileExtension)
		writeStringToHere(pythonSource, pathFilenameModule)

	astImportFromClassNewInPythonVersion = ast.ImportFrom('mapFolding', [ast.alias(pleasedonotcrashwhileimportingtypes) for pleasedonotcrashwhileimportingtypes in list_astDOTStuPyd], 0)
	keywordArgumentsIdentifier: ast_Identifier = 'keywordArguments'
	moduleIdentifierPrefix: str = '_tool_'
	overloadName = ast.Name('overload', ast.Load())
	staticmethodName = ast.Name('staticmethod', ast.Load())
	typing_TypeAliasName: ast.expr = cast(ast.expr, ast.Name('typing_TypeAlias', ast.Load()))

	beClassDef = ast.ClassDef(name='be', bases=[], keywords=[], body=[], decorator_list=[])
	DOTClassDef = ast.ClassDef(name='DOT', bases=[], keywords=[], body=[], decorator_list=[])
	MakeClassDef = ast.ClassDef(name='Make', bases=[], keywords=[], body=[], decorator_list=[])
	grabClassDef = ast.ClassDef(name='grab', bases=[], keywords=[], body=[], decorator_list=[])

	dictionary_astClassAnnotations: dict[ast_Identifier, ast.Name | ast.Attribute] = MakeDictionaryOf_astClassAnnotations(astStubFile).getDictionary()
	Z0Z_dictionaryDeconstructedAttributes: dict[ast_Identifier, dict[str, list[ast_Identifier | str_nameDOTname]]] = {}
	# Track original type annotations for each attribute to be used for return types
	Z0Z_attributeTypeAnnotations: dict[ast_Identifier, dict[str, ast.expr]] = {}

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
		ClassDefNameOrAttribute = dictionary_astClassAnnotations[ClassDefIdentifier]
		keywordArguments_ast_arg = ast.arg(arg=keywordArgumentsIdentifier, annotation=ast.Name(id='int', ctx=ast.Load()))
		keywordArguments_ast_keyword = ast.keyword(value=ast.Name(id=keywordArgumentsIdentifier, ctx=ast.Load()))
		# Create the ClassDef and add directly to the body of the class.
		beClassDef.body.append(ast.FunctionDef(name=ClassDefIdentifier
			, args=ast.arguments(posonlyargs=[]
			, args=[ast.arg(arg='node', annotation=ast.Name('ast.AST'))], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[ast.Return(value=ast.Call(func=ast.Name('isinstance'), args=[ast.Name('node'), ClassDefNameOrAttribute], keywords=[]))], decorator_list=[staticmethodName], type_comment=None, returns=ast.Subscript(value=ast.Name('TypeGuard'), slice=ClassDefNameOrAttribute, ctx=ast.Load())))

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
		match ClassDefIdentifier:
			case 'Module' | 'Interactive' | 'FunctionType' | 'Expression':
				keywordArguments_ast_arg = None
				keywordArguments_ast_keyword = None
			case _:
				pass
		MakeClassDef.body.append(ast.FunctionDef(name=ClassDefIdentifier
			, args=ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=keywordArguments_ast_arg, defaults=[])
			, body=[ast.Return(value=ast.Call(func=ClassDefNameOrAttribute, args=[]
							, keywords=[keywordArguments_ast_keyword] if keywordArguments_ast_keyword else []
							))]
			, decorator_list=[staticmethodName], type_comment=None, returns=ClassDefNameOrAttribute))

		SubstituteTyping_TypeAlias().visit(ImaClassDef)
		for attributeIdentifier in listAttributes:
			for subnode in ast.walk(ImaClassDef):
				if isinstance(subnode, ast.AnnAssign) and isinstance(subnode.target, ast.Name) and subnode.target.id == attributeIdentifier:
					# TODO Change some list to Sequence
					subnode_annotation = ChangeName2Attribute(dictionary_astClassAnnotations).visit(subnode.annotation)
					if attributeIdentifier not in Z0Z_dictionaryDeconstructedAttributes:
						Z0Z_dictionaryDeconstructedAttributes[attributeIdentifier] = {}
						Z0Z_attributeTypeAnnotations[attributeIdentifier] = {}
					subnode_annotation_str = ''.join([letter for letter in ast.unparse(subnode.annotation).replace('ast','').replace('|','Or') if letter in ascii_letters])
					Z0Z_dictionaryDeconstructedAttributes[attributeIdentifier].setdefault(subnode_annotation_str, []).append(node.name)
					Z0Z_attributeTypeAnnotations[attributeIdentifier][subnode_annotation_str] = subnode_annotation

					match ClassDefIdentifier:
						case 'Attribute':
							if cast(ast.FunctionDef, MakeClassDef.body[-1]).name == ClassDefIdentifier:
								MakeClassDef.body.pop(-1)
							MakeClassDef.body.append(MakeAttributeFunctionDef)
							continue
						case 'Import':
							if cast(ast.FunctionDef, MakeClassDef.body[-1]).name == ClassDefIdentifier:
								MakeClassDef.body.pop(-1)
							MakeClassDef.body.append(MakeImportFunctionDef)
							continue
						case _:
							pass

					match attributeIdentifier:
						case 'args':
							if 'list' in subnode_annotation_str:
								cast(ast.FunctionDef, MakeClassDef.body[-1]).args.defaults.append(ast.List([]))
						case 'asname':
							attributeIdentifier = 'asName'
							cast(ast.FunctionDef, MakeClassDef.body[-1]).args.defaults.append(ast.Constant(None))
						case 'bases':
							cast(ast.FunctionDef, MakeClassDef.body[-1]).args.defaults.append(ast.List([]))
						case 'ctx':
							attributeIdentifier = 'context'
							cast(ast.FunctionDef, MakeClassDef.body[-1]).args.defaults.append(ast.Call(ast.Attribute(ast.Name('ast', ctx=ast.Load()), attr='Load', ctx=ast.Load())))
						case 'decorator_list':
							cast(ast.FunctionDef, MakeClassDef.body[-1]).args.defaults.append(ast.List([]))
						case 'defaults':
							cast(ast.FunctionDef, MakeClassDef.body[-1]).args.defaults.append(ast.List([]))
						case 'func':
							attributeIdentifier = 'callee'
						case 'kind':
							cast(ast.arg, cast(ast.FunctionDef, MakeClassDef.body[-1]).args.kwarg).annotation = ast.Name('intORstr', ctx=ast.Load())
							continue
						case 'keywords':
							attributeIdentifier = 'list_keyword'
							cast(ast.FunctionDef, MakeClassDef.body[-1]).args.defaults.append(ast.List([]))
						case 'kw_defaults':
							cast(ast.FunctionDef, MakeClassDef.body[-1]).args.defaults.append(ast.List([ast.Constant(None)]))
						case 'kwarg':
							cast(ast.FunctionDef, MakeClassDef.body[-1]).args.defaults.append(ast.Constant(None))
						case 'kwonlyargs':
							cast(ast.FunctionDef, MakeClassDef.body[-1]).args.defaults.append(ast.List([]))
						case 'level':
							cast(ast.Call, cast(ast.Return, cast(ast.FunctionDef, MakeClassDef.body[-1]).body[0]).value).keywords.append(ast.keyword(attributeIdentifier, ast.Constant(0)))
							continue
						case 'names':
							if ClassDefIdentifier == 'ImportFrom':
								attributeIdentifier = 'list_alias'
						case 'orelse':
							attributeIdentifier = 'orElse'
							if 'list' in subnode_annotation_str:
								cast(ast.FunctionDef, MakeClassDef.body[-1]).args.defaults.append(ast.List([]))
						case 'posonlyargs':
							cast(ast.FunctionDef, MakeClassDef.body[-1]).args.defaults.append(ast.List([]))
						case 'returns':
							match ClassDefIdentifier:
								case 'FunctionType':
									pass
								case _:
									cast(ast.FunctionDef, MakeClassDef.body[-1]).args.defaults.append(ast.Constant(None))
						case 'simple':
							cast(ast.Call, cast(ast.Return, cast(ast.FunctionDef, MakeClassDef.body[-1]).body[0]).value).keywords.append(ast.keyword(attributeIdentifier
									, ast.Call(func=ast.Name(id='int', ctx=ast.Load()), args=[ast.Call(func=ast.Name(id='isinstance', ctx=ast.Load()), args=[ast.Name(id='target', ctx=ast.Load()), ast.Attribute(value=ast.Name(id='ast', ctx=ast.Load()), attr='Name', ctx=ast.Load())])])))
							continue
						case 'type_comment':
							cast(ast.arg, cast(ast.FunctionDef, MakeClassDef.body[-1]).args.kwarg).annotation = ast.Name('intORstr', ctx=ast.Load())
							continue
						case 'type_ignores':
							cast(ast.FunctionDef, MakeClassDef.body[-1]).args.defaults.append(ast.List([]))
						case 'type_params':
							match ClassDefIdentifier:
								case 'AsyncFunctionDef' | 'FunctionDef':
									cast(ast.arg, cast(ast.FunctionDef, MakeClassDef.body[-1]).args.kwarg).annotation = ast.Name('intORstrORtype_params', ctx=ast.Load())
									continue
								case 'ClassDef':
									cast(ast.arg, cast(ast.FunctionDef, MakeClassDef.body[-1]).args.kwarg).annotation = ast.Name('intORtype_params', ctx=ast.Load())
									continue
								case _:
									pass
						case 'vararg':
							cast(ast.FunctionDef, MakeClassDef.body[-1]).args.defaults.append(ast.Constant(None))
						case _:
							pass
					cast(ast.FunctionDef, MakeClassDef.body[-1]).args.args.append(ast.arg(arg=attributeIdentifier, annotation=subnode_annotation))
					cast(ast.Call, cast(ast.Return, cast(ast.FunctionDef, MakeClassDef.body[-1]).body[0]).value).args.append(ast.Name(attributeIdentifier, ctx=ast.Load()))

	beClassDef.body.sort(key=lambda astFunctionDef: astFunctionDef.name.lower()) # pyright: ignore[reportAttributeAccessIssue, reportUnknownLambdaType, reportUnknownMemberType]
	MakeClassDef.body.sort(key=lambda astFunctionDef: astFunctionDef.name.lower()) # pyright: ignore[reportAttributeAccessIssue, reportUnknownLambdaType, reportUnknownMemberType]

	astTypesModule = ast.Module(
		body=[ast.Expr(ast.Constant(docstringWarning))
			, astImportFromClassNewInPythonVersion
			, ast.ImportFrom('typing', [ast.alias('Any'), ast.alias('TypeAlias', 'typing_TypeAlias')], 0)
			, ast.Import([ast.alias('ast')])
			, *handmadeTypeAlias_astTypes
			]
		, type_ignores=[]
		)

	listAttributeIdentifier: list[ast_Identifier] = list(Z0Z_dictionaryDeconstructedAttributes.keys())
	listAttributeIdentifier.sort(key=lambda attributeIdentifier: attributeIdentifier.lower())

	# Build mappings from attribute to TypeAlias and original types
	attribute2TypeAlias: dict[str, list[ast.expr]] = {}
	attribute_to_types: dict[str, list[ast.expr]] = {}

	for attributeIdentifier in listAttributeIdentifier:
		hasDOTIdentifier: ast_Identifier = 'hasDOT' + attributeIdentifier
		attribute2TypeAlias[attributeIdentifier] = []
		attribute_to_types[attributeIdentifier] = []
		for subnode_annotation, listClassDefIdentifier in Z0Z_dictionaryDeconstructedAttributes[attributeIdentifier].items():
			hasDOT_subnode_name = ast.Name(hasDOTIdentifier + '_' + subnode_annotation.replace('list', 'list_'), ast.Load()) if len(Z0Z_dictionaryDeconstructedAttributes[attributeIdentifier]) > 1 else ast.Name(hasDOTIdentifier, ast.Load())
			attribute2TypeAlias[attributeIdentifier].append(hasDOT_subnode_name)
			type_annotation = Z0Z_attributeTypeAnnotations[attributeIdentifier][subnode_annotation]
			attribute_to_types[attributeIdentifier].append(type_annotation)

	for attributeIdentifier in listAttributeIdentifier:
		hasDOTIdentifier: ast_Identifier = 'hasDOT' + attributeIdentifier
		hasDOTName_Store: ast.Name = ast.Name(hasDOTIdentifier, ast.Store())
		hasDOTName_Load: ast.Name = ast.Name(hasDOTIdentifier, ast.Load())
		list_hasDOTName_subnode_annotation: list[ast.Name] = []
		dictionaryAnnotations = Z0Z_dictionaryDeconstructedAttributes[attributeIdentifier]
		for subnode_annotation, listClassDefIdentifier in dictionaryAnnotations.items():
			astAnnAssignValue = None
			if len(listClassDefIdentifier) == 1:
				astAnnAssignValue = dictionary_astClassAnnotations[listClassDefIdentifier[0]]
			else:
				astAnnAssignValue = dictionary_astClassAnnotations[listClassDefIdentifier[0]]
				for ClassDefIdentifier in listClassDefIdentifier[1:]:
					astAnnAssignValue = ast.BinOp(left=astAnnAssignValue, op=ast.BitOr(), right=dictionary_astClassAnnotations[ClassDefIdentifier])
			if len(dictionaryAnnotations) == 1:
				astTypesModule.body.append(ast.AnnAssign(hasDOTName_Store, typing_TypeAliasName, astAnnAssignValue, 1))
			else:
				list_hasDOTName_subnode_annotation.append(ast.Name(hasDOTIdentifier + '_' + subnode_annotation.replace('list', 'list_'), ast.Store()))
				astTypesModule.body.append(ast.AnnAssign(list_hasDOTName_subnode_annotation[-1], typing_TypeAliasName, astAnnAssignValue, 1))				# For overloaded methods, use the specific attribute type
				DOTClassDef.body.append(ast.FunctionDef(name=attributeIdentifier,
						args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='node', annotation=ast.Name(list_hasDOTName_subnode_annotation[-1].id, ast.Load()))], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
					body=[ast.Expr(value=ast.Constant(value=Ellipsis))],
					decorator_list=[staticmethodName, overloadName],
					# Use the appropriate type annotation for the return value
					returns=Z0Z_attributeTypeAnnotations[attributeIdentifier][subnode_annotation] if attributeIdentifier in Z0Z_attributeTypeAnnotations and subnode_annotation in Z0Z_attributeTypeAnnotations[attributeIdentifier] else astAnnAssignValue
				))
		if list_hasDOTName_subnode_annotation:
			astAnnAssignValue = list_hasDOTName_subnode_annotation[0]
			for index in range(1, len(list_hasDOTName_subnode_annotation)):
				astAnnAssignValue = ast.BinOp(left=astAnnAssignValue, op=ast.BitOr(), right=list_hasDOTName_subnode_annotation[index])
			astTypesModule.body.append(ast.AnnAssign(hasDOTName_Store, typing_TypeAliasName, astAnnAssignValue, 1))		# Create a function to get the attribute with the correct return type
		# Replace None with the appropriate type for each attribute
		attribute_return_type = None
		if attributeIdentifier in attribute2TypeAlias:
			if len(attribute2TypeAlias[attributeIdentifier]) == 1:
				attribute_return_type = attribute_to_types[attributeIdentifier][0]
			else:
				# Create a union of all possible return types
				attribute_return_type = attribute_to_types[attributeIdentifier][0]
				for i in range(1, len(attribute_to_types[attributeIdentifier])):
					attribute_return_type = ast.BinOp(
						left=attribute_return_type,
						op=ast.BitOr(),
						right=attribute_to_types[attributeIdentifier][i]
					)

		DOTClassDef.body.append(ast.FunctionDef(name=attributeIdentifier
				, args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='node', annotation=hasDOTName_Load)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
				, body=[ast.Return(value=ast.Attribute(value=ast.Name('node', ast.Load()), attr=attributeIdentifier, ctx=ast.Load()))]
				, decorator_list=[staticmethodName]
				, returns=attribute_return_type
			))

		# For grab class, use the appropriate attribute type
		attribute_type = None
		if attributeIdentifier in attribute_to_types:
			if len(attribute_to_types[attributeIdentifier]) == 1:
				attribute_type = attribute_to_types[attributeIdentifier][0]
			else:
				attribute_type = attribute_to_types[attributeIdentifier][0]
				for i in range(1, len(attribute_to_types[attributeIdentifier])):
					attribute_type = ast.BinOp(
						left=attribute_type,
						op=ast.BitOr(),
						right=attribute_to_types[attributeIdentifier][i]
					)
		else:
			attribute_type = ast.Name('Any', ast.Load())

		grabClassDef.body.append(ast.FunctionDef(name=attributeIdentifier + 'Attribute'
			, args=ast.arguments(posonlyargs=[]
				, args=[ast.arg('action'
					, annotation=ast.Subscript(ast.Name('Callable', ast.Load())
						, slice=ast.Tuple(elts=[
							ast.List(elts=[attribute_type or ast.Name('Any', ast.Load())], ctx=ast.Load())
							,   attribute_type or ast.Name('Any', ast.Load())]
						, ctx=ast.Load()), ctx=ast.Load()))]
				, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
			, body=[ast.FunctionDef(name='workhorse',
						args=ast.arguments(args=[ast.arg('node', hasDOTName_Load)]),
					body=[ast.Assign(targets=[ast.Attribute(ast.Name('node', ast.Load()), attr=attributeIdentifier, ctx=ast.Store())],
						value=ast.Call(ast.Name('action', ast.Load()), args=[ast.Attribute(ast.Name('node', ast.Load()), attr=attributeIdentifier, ctx=ast.Load())]))
						, ast.Return(ast.Name('node', ast.Load()))],
						returns=hasDOTName_Load),
			ast.Return(ast.Name('workhorse', ctx=ast.Load()))]
			, decorator_list=[staticmethodName], type_comment=None
			, returns=ast.Subscript(ast.Name('Callable', ast.Load()), ast.Tuple([ast.List([hasDOTName_Load], ast.Load()), hasDOTName_Load], ast.Load()), ast.Load())))

	writeModule(astTypesModule, '_astTypes')

	beClassDef.body.insert(0, ast.Expr(value=ast.Constant(value=beClassDefDocstring)))
	DOTClassDef.body.insert(0, ast.Expr(value=ast.Constant(value=DOTClassDefDocstring)))
	grabClassDef.body.insert(0, ast.Expr(value=ast.Constant(value=grabClassDefDocstring)))
	MakeClassDef.body.insert(0, ast.Expr(value=ast.Constant(value=MakeClassDefDocstring)))

	grabClassDef.body.extend(handmadeMethods_grab)

	writeModule(ast.Module(
		body=[ast.Expr(ast.Constant(docstringWarning))
			, astImportFromClassNewInPythonVersion
			, ast.ImportFrom('typing', [ast.alias('TypeGuard')], 0)
			, ast.Import([ast.alias('ast')])
			, beClassDef
			],
		type_ignores=[]
		)
		, moduleIdentifierPrefix + beClassDef.name)

	writeModule(ast.Module(
		body=[ast.Expr(ast.Constant(docstringWarning))
			, astImportFromClassNewInPythonVersion
			, ast.ImportFrom('mapFolding.someAssemblyRequired', [ast.alias('ast_Identifier'), ast.alias('ast_expr_Slice')], 0)
			, ast.ImportFrom('mapFolding.someAssemblyRequired._astTypes', [ast.alias('*')], 0)
			, ast.ImportFrom('typing', [ast.alias(identifier) for identifier in ['Any', 'Literal', 'overload']], 0)
			, ast.Import([ast.alias('ast')])
			# TODO but idk what
			, ast.Expr(ast.Constant('# ruff: noqa: F405'))
			, DOTClassDef
			],
		type_ignores=[]
		)
		, moduleIdentifierPrefix + DOTClassDef.name)

	writeModule(ast.Module(
		body=[ast.Expr(ast.Constant(docstringWarning))
			, ast.ImportFrom('collections.abc', [ast.alias('Callable')], 0)
			, astImportFromClassNewInPythonVersion
			, ast.ImportFrom('mapFolding.someAssemblyRequired', [ast.alias(identifier) for identifier in ['ast_Identifier', 'ast_expr_Slice', 'NodeORattribute', 'ImaCallToName']], 0)
			, ast.ImportFrom('mapFolding.someAssemblyRequired._astTypes', [ast.alias('*')], 0)
			, ast.ImportFrom('typing', [ast.alias('Any'), ast.alias('Literal')], 0)
			, ast.Import([ast.alias('ast')])
			# TODO but idk what
			, ast.Expr(ast.Constant('# ruff: noqa: F405'))
			, grabClassDef
			],
		type_ignores=[]
		)
		, moduleIdentifierPrefix + grabClassDef.name)

	writeModule(ast.Module(
		body=[ast.Expr(ast.Constant(docstringWarning))
			, astImportFromClassNewInPythonVersion
			, ast.ImportFrom('mapFolding.someAssemblyRequired', [ast.alias(identifier) for identifier in ['ast_Identifier', 'ast_expr_Slice', 'intORstr', 'intORstrORtype_params', 'intORtype_params', 'str_nameDOTname']], 0)
			, ast.ImportFrom('typing', [ast.alias('Any'), ast.alias('Literal')], 0)
			, ast.Import([ast.alias('ast')])
			, MakeClassDef
			],
		type_ignores=[]
		)
		, moduleIdentifierPrefix + MakeClassDef.name)
