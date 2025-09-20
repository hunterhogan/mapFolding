from collections.abc import Callable
from mapFolding.dataBaskets import MapFoldingState
from typing import Any
import astToolkit
import dataclasses
import functools
import inspect

r"""Notes
Goal: create a decorator, `unRePackMapFoldingState`, that will unpack `MapFoldingState`, pass only the parameters in the decorated function, receive the
values returned by the function, and repack `MapFoldingState`.

You must use dynamic programming. If the datatype or the field name changes, for example, that should not affect the decorator.

To use in, for example, C:\apps\mapFolding\mapFolding\syntheticModules\A007822\theorem2Numba.py. Analogous to
`Z0Z_tools.waveformSpectrogramWaveform`, see
https://github.com/hunterhogan/Z0Z_tools/blob/2c393c2831382dfe6f3e742cf56db39e71126cbc/Z0Z_tools/ioAudio.py

For examples of manipulating `dataclasses`, see also:
C:\apps\mapFolding\mapFolding\someAssemblyRequired\_toolkitContainers.py and
C:\apps\mapFolding\mapFolding\someAssemblyRequired\transformationTools.py

- `TypeVar` may be useful.

- `dataclasses` has inspection tools.

- `return MapFoldingState(...` Check if the field is init=True

Prototype. Eventual home will probably be "beDry.py".
"""

def unRePackMapFoldingState[CallableTargetType: Callable[..., Any]](callableTarget: CallableTargetType) -> Callable[[MapFoldingState], MapFoldingState]:
	signatureTargetFunction: inspect.Signature = inspect.signature(callableTarget)
	parametersTargetFunction: list[str] = list(signatureTargetFunction.parameters.keys())

	fieldsMapFoldingState: tuple[dataclasses.Field[Any], ...] = dataclasses.fields(MapFoldingState)

	fieldsInitializable: dict[str, dataclasses.Field[Any]] = {field.name: field for field in fieldsMapFoldingState if field.init}

	@functools.wraps(callableTarget)
	def decoratedFunction(mapFoldingStateInstance: MapFoldingState, **additionalKeywordArguments: Any) -> MapFoldingState:
		dataclassAsDict: dict[str, Any] = dataclasses.asdict(mapFoldingStateInstance)

		argumentsForTargetFunction: list[Any] = []
		for parameterName in parametersTargetFunction:
			if parameterName in dataclassAsDict:
				argumentsForTargetFunction.append(dataclassAsDict[parameterName])
			elif parameterName in additionalKeywordArguments:
				argumentsForTargetFunction.append(additionalKeywordArguments[parameterName])
			else:
				errorMessage = f"Parameter '{parameterName}' not found in MapFoldingState or additional arguments"
				raise ValueError(errorMessage)

		returnedFromTargetFunction: Any = callableTarget(*argumentsForTargetFunction)

		argumentsForMapFoldingStateConstructor: dict[str, Any] = {fieldName: dataclassAsDict[fieldName] for fieldName in fieldsInitializable}

		if len(parametersTargetFunction) == 1:
			singleParameterName: str = parametersTargetFunction[0]
			if singleParameterName in fieldsInitializable:
				argumentsForMapFoldingStateConstructor[singleParameterName] = returnedFromTargetFunction
		elif isinstance(returnedFromTargetFunction, tuple) and len(returnedFromTargetFunction) == len(parametersTargetFunction):
			updatedFieldsFromReturn: dict[str, Any] = {parameterName: returnedValue for parameterName, returnedValue in zip(parametersTargetFunction, returnedFromTargetFunction, strict=True) if parameterName in fieldsInitializable}
			argumentsForMapFoldingStateConstructor.update(updatedFieldsFromReturn)

		return MapFoldingState(**argumentsForMapFoldingStateConstructor)

	return decoratedFunction
