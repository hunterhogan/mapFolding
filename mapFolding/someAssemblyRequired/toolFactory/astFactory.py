from mapFolding import The, writeStringToHere
from os import PathLike
from pathlib import PurePath, PurePosixPath
from typing import TypeAlias as typing_TypeAlias, cast
import ast
import inspect
import typing
import types
import re

ast_Identifier: typing_TypeAlias = str

def getSetAstClassDeprecatedDynamic() -> set[ast_Identifier]:
	setAstClassDeprecatedDynamic: set[ast_Identifier] = set()
	for identifierAstClass, typeAstClass in inspect.getmembers(ast, inspect.isclass):
		if not issubclass(typeAstClass, ast.AST):
			continue
		docstringAstClass: ast_Identifier | None = typeAstClass.__doc__
		if docstringAstClass and 'deprecated' in docstringAstClass.lower():
			setAstClassDeprecatedDynamic.add(identifierAstClass)
	return setAstClassDeprecatedDynamic

def getListAstClassIdentifiers(
	setAstClassDeprecated: set[ast_Identifier],
) -> list[ast_Identifier]:
	listAstClassIdentifiers: list[ast_Identifier] = []
	for identifierAstClass, typeAstClass in inspect.getmembers(ast, inspect.isclass):
		if not issubclass(typeAstClass, ast.AST):
			continue
		if identifierAstClass in setAstClassDeprecated:
			continue
		if identifierAstClass.startswith('AST') or identifierAstClass.startswith('_'):
			continue
		listAstClassIdentifiers.append(identifierAstClass)
	return sorted(listAstClassIdentifiers, key=lambda stringAstClass: stringAstClass.lower())

def getStringClassBe(listAstClassIdentifiers: list[ast_Identifier]) -> str:
	listLine: list[str] = []
	listLine.append('class be:')
	docstring: str ="""
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

	listLine.append('    """')
	listLine.append(docstring)
	listLine.append('    """')
	for identifierAstClass in listAstClassIdentifiers:
		listLine.append('    @staticmethod')
		listLine.append(f'    def {identifierAstClass}(nodeAst: ast.AST) -> TypeGuard[ast.{identifierAstClass}]:')
		listLine.append(f'        return isinstance(nodeAst, ast.{identifierAstClass})')
		listLine.append('')
	return '\n'.join(listLine)

def writeModuleBe(pathFilenameTarget: PathLike[str] | PurePath, setAstClassDeprecated: set[ast_Identifier], list_astDOTStuPyd: list[ast_Identifier]) -> None:
	listAstClassIdentifiers = getListAstClassIdentifiers(setAstClassDeprecated)
	stringHeader: str = '"""This file is generated automatically, so changes to this file will be lost."""\nfrom typing import TypeGuard\nimport ast\n'
	stringHeader = stringHeader + f"from mapFolding import {', '.join(list_astDOTStuPyd)}\n"
	stringCode: str = stringHeader + '\n' + getStringClassBe(listAstClassIdentifiers) + '\n'
	for pyDOTwhy in list_astDOTStuPyd:
		astClass = pyDOTwhy.replace('DOT', '.')
		stringCode = stringCode.replace(astClass, pyDOTwhy)
	writeStringToHere(stringCode, pathFilenameTarget)

def getStringTypeNameForIdentifier(attributeType: object) -> str:
	if attributeType is bool:
		return 'bool'
	if attributeType is int:
		return 'int'
	if attributeType is str:
		return 'str'
	if attributeType is None or attributeType is type(None):
		return 'None'
	if attributeType is typing.Any or attributeType is object:
		return 'Any'
	if hasattr(attributeType, '__origin__') and getattr(attributeType, '__origin__', None) is list:
		arg = getattr(attributeType, '__args__', [None])[0]
		return 'list_' + getStringTypeNameForIdentifier(arg)
	# Handle typing.Union and | (PEP 604)
	if (hasattr(attributeType, '__origin__') and getattr(attributeType, '__origin__', None) is typing.Union # pyright: ignore[reportDeprecated]
		or hasattr(types, 'UnionType') and isinstance(attributeType, types.UnionType)):
		args = getattr(attributeType, '__args__', None)
		if args is not None:
			names = [getStringTypeNameForIdentifier(arg) for arg in args]
			return 'OR'.join(names)
	if isinstance(attributeType, str):
		return attributeType.replace("'", "")
	if hasattr(attributeType, '__name__') and getattr(attributeType, '__name__', None):
		return getattr(attributeType, '__name__')
	if hasattr(attributeType, '_name') and getattr(attributeType, '_name', None):
		return getattr(attributeType, '_name')
	# Fallback: try to parse ast types
	stringType = str(attributeType)
	if 'ast.' in stringType:
		match = re.search(r"ast\.([A-Za-z0-9_]+)", stringType)
		if match:
			return 'ast.' + match.group(1)
	if 'str' in stringType:
		return 'str'
	if 'int' in stringType:
		return 'int'
	if 'bool' in stringType:
		return 'bool'
	if 'NoneType' in stringType:
		return 'None'
	if 'Any' in stringType:
		return 'Any'
	return stringType.replace('typing.', '').replace('ast.', 'ast.').replace("'", '').replace('[', '').replace(']', '').replace(' ', '')

def getMappingAttributeToClassSetRich() -> dict[ast_Identifier, dict[str, dict[str, object]]]:
	mapping: dict[ast_Identifier, dict[str, dict[str, object]]] = {}
	for identifierAstClass, typeAstClass in inspect.getmembers(ast, inspect.isclass):
		if not issubclass(typeAstClass, ast.AST):
			continue
		annotations = getattr(typeAstClass, '__annotations__', {})
		if identifierAstClass == 'MatchSingleton':
			annotations = {'value': bool | None}
		for attributeName, attributeType in annotations.items():
			typeName = getStringTypeNameForIdentifier(attributeType)
			if attributeName not in mapping:
				mapping[attributeName] = {}
			if typeName not in mapping[attributeName]:
				mapping[attributeName][typeName] = {'classes': set(), 'class2type': {}}
			classSet = cast(set[str], mapping[attributeName][typeName]['classes'])
			class2type = cast(dict[str, str], mapping[attributeName][typeName]['class2type'])
			classSet.add(identifierAstClass)
			class2type[identifierAstClass] = typeName
	return mapping

def getStringTypeAliasDefinitions(mappingAttributeToTypeToClassSet: dict[ast_Identifier, dict[str, set[ast_Identifier]]]) -> str:
	def resolve_type(typeName: str) -> str:
		if typeName == 'ast_Identifier':
			return 'ast_Identifier'
		if typeName == 'Any':
			return 'Any'
		if typeName == 'bool':
			return 'bool'
		if typeName == 'int':
			return 'int'
		if typeName == 'str':
			return 'str'
		if typeName == 'None':
			return 'None'
		if typeName.startswith('list_'):
			listType = typeName[5:]
			resolved = resolve_type(listType)
			return f'list[{resolved}]'
		if 'OR' in typeName:
			return ' | '.join(resolve_type(t) for t in typeName.split('OR'))
		if typeName.startswith('ast.'):
			return typeName
		return f'ast.{typeName}'
	listLine: list[str] = []
	for attributeName in sorted(mappingAttributeToTypeToClassSet.keys()):
		typeAliasIdentifier = f'hasDOT{attributeName}'
		mappingTypeToClassSet = mappingAttributeToTypeToClassSet[attributeName]
		if len(mappingTypeToClassSet) > 1:
			listSubtypeAliasIdentifier: list[str] = []
			for typeName in sorted(mappingTypeToClassSet.keys()):
				setClassIdentifiers = mappingTypeToClassSet[typeName]
				subtypeAliasIdentifier = f'{typeAliasIdentifier}_{typeName}'
				listSubtypeAliasIdentifier.append(subtypeAliasIdentifier)
				classUnion = ' | '.join(f'ast.{classIdentifier}' for classIdentifier in sorted(setClassIdentifiers))
				listLine.append(f'{subtypeAliasIdentifier}: typing_TypeAlias = {classUnion}')
			listLine.append(f'{typeAliasIdentifier}: typing_TypeAlias = {" | ".join(listSubtypeAliasIdentifier)}')
		else:
			for typeName in sorted(mappingTypeToClassSet.keys()):
				setClassIdentifiers = mappingTypeToClassSet[typeName]
				classUnion = ' | '.join(f'ast.{classIdentifier}' for classIdentifier in sorted(setClassIdentifiers))
				listLine.append(f'{typeAliasIdentifier}: typing_TypeAlias = {classUnion}')
	return '\n'.join(listLine)

def writeModuleDOT(pathFilename_DOT: PathLike[str] | PurePath, mappingAttributeToTypeToClassSetRich: dict[ast_Identifier, dict[str, dict[str, object]]], list_astDOTStuPyd: list[ast_Identifier]) -> None:
	from typing import Any, cast
	lines: list[str] = []
	lines.append('from typing import overload, Any')
	lines.append('from mapFolding.someAssemblyRequired import ast_Identifier')
	lines.append(f"from mapFolding import {', '.join(list_astDOTStuPyd)}")
	lines.append('import ast')
	lines.append('')
	lines.append('class DOT:')
	for attributeName in sorted(mappingAttributeToTypeToClassSetRich.keys()):
		typeMap = mappingAttributeToTypeToClassSetRich[attributeName]
		inputTypes: list[str] = []
		returnTypes: list[str] = []
		overloads: list[tuple[str, str]] = []
		for typeName in sorted(typeMap.keys()):
			classSet = cast(set[str], typeMap[typeName]['classes'])
			inputType = ' | '.join(f'ast.{className}' for className in sorted(classSet))
			# Return type logic
			def resolve_type(typeName: str) -> str:
				if typeName == 'ast_Identifier':
					return 'ast_Identifier'
				if typeName == 'Any':
					return 'Any'
				if typeName == 'bool':
					return 'bool'
				if typeName == 'int':
					return 'int'
				if typeName == 'str':
					return 'str'
				if typeName == 'None':
					return 'None'
				if typeName.startswith('list_'):
					listType = typeName[5:]
					resolved = resolve_type(listType)
					return f'list[{resolved}]'
				if 'OR' in typeName:
					return ' | '.join(resolve_type(t) for t in typeName.split('OR'))
				if typeName.startswith('ast.'):
					return typeName
				return f'ast.{typeName}'
			returnType = resolve_type(typeName)
			overloads.append((inputType, returnType))
			inputTypes.append(inputType)
			returnTypes.append(returnType)
		overloads = list(dict.fromkeys(overloads))
		inputTypes = list(dict.fromkeys(inputTypes))
		returnTypes = list(dict.fromkeys(returnTypes))
		if len(overloads) > 1:
			for inputType, returnType in overloads:
				lines.append('    @staticmethod')
				lines.append('    @overload')
				lines.append(f'    def {attributeName}(node: {inputType}) -> {returnType}: ...')
			lines.append('    @staticmethod')
			lines.append(f'    def {attributeName}(node: {" | ".join(inputTypes)}) -> {" | ".join(returnTypes)}:')
			lines.append(f'        return node.{attributeName}')
			lines.append('')
		else:
			inputType, returnType = overloads[0]
			lines.append('    @staticmethod')
			lines.append(f'    def {attributeName}(node: {inputType}) -> {returnType}:')
			lines.append(f'        return node.{attributeName}')
			lines.append('')
	stringCode = '\n'.join(lines)
	for pyDOTwhy in list_astDOTStuPyd:
		astClass = pyDOTwhy.replace('DOT', '.')
		stringCode = stringCode.replace(astClass, pyDOTwhy)
	writeStringToHere(stringCode, pathFilename_DOT)

def writeModules_astTypesAndDOT(pathFilename_astTypes: PathLike[str] | PurePath, pathFilename_DOT: PathLike[str] | PurePath, list_astDOTStuPyd: list[ast_Identifier]) -> None:
	mappingAttributeToTypeToClassSetRich = getMappingAttributeToClassSetRich()
	# Type aliases module
	mappingForTypeAliases = {k: {t: cast(set[str], v['classes']) for t, v in mappingAttributeToTypeToClassSetRich[k].items()} for k in mappingAttributeToTypeToClassSetRich}  # type: ignore
	stringHeader = f"from mapFolding import {', '.join(list_astDOTStuPyd)}\nfrom typing import TypeAlias as typing_TypeAlias\nimport ast\n\n"
	stringTypeAliases = getStringTypeAliasDefinitions(mappingForTypeAliases)
	for pyDOTwhy in list_astDOTStuPyd:
		astClass = pyDOTwhy.replace('DOT', '.')
		stringTypeAliases = stringTypeAliases.replace(astClass, pyDOTwhy)
	stringCode = stringHeader + '\n' + stringTypeAliases + '\n'
	writeStringToHere(stringCode, pathFilename_astTypes)
	# DOT module
	writeModuleDOT(pathFilename_DOT, mappingAttributeToTypeToClassSetRich, list_astDOTStuPyd)

if __name__ == '__main__':
	logicalPathInfixHARDCODED = 'someAssemblyRequired'
	moduleIdentifier_astTypesHARDCODED = '_astTypes'
	moduleIdentifier_beHARDCODED = '_tool_be'
	moduleIdentifier_DOTHARDCODED = '_tool_DOT'
	setAstClassDeprecatedHARDCODED: set[ast_Identifier] = {
		'Num', 'Str', 'Bytes', 'NameConstant', 'Ellipsis',
		'Index', 'ExtSlice', 'Suite', 'AugLoad', 'AugStore', 'Param',
	}

	list_astDOTStuPydHARDCODED: list[ast_Identifier] = ['astDOTParamSpec', 'astDOTTryStar', 'astDOTTypeAlias', 'astDOTTypeVar', 'astDOTTypeVarTuple']
	list_astDOTStuPyd = list_astDOTStuPydHARDCODED.copy()

	logicalPathInfix = logicalPathInfixHARDCODED
	moduleIdentifier_be = moduleIdentifier_beHARDCODED
	pathFilename_be = PurePosixPath(The.pathPackage, logicalPathInfix, moduleIdentifier_be + The.fileExtension)

	moduleIdentifier_astTypes = moduleIdentifier_astTypesHARDCODED
	pathFilename_astTypes = PurePosixPath(The.pathPackage, logicalPathInfix, moduleIdentifier_astTypes + The.fileExtension)

	moduleIdentifier_DOT = moduleIdentifier_DOTHARDCODED
	pathFilename_DOT = PurePosixPath(The.pathPackage, logicalPathInfix, moduleIdentifier_DOT + The.fileExtension)

	setAstClassDeprecated: set[ast_Identifier] = setAstClassDeprecatedHARDCODED | getSetAstClassDeprecatedDynamic()

	writeModuleBe(pathFilename_be, setAstClassDeprecated, list_astDOTStuPyd)
	writeModules_astTypesAndDOT(pathFilename_astTypes, pathFilename_DOT, list_astDOTStuPyd)
