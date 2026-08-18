#=SIN= Pyright suppression: Numba extension hooks lack stable public annotations for their low-level code-generation types.
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
#=SIN= Ruff suppression: Numba StructRef integration requires its private payload utilities and native value accessor.
# ruff: file-ignore[private-member-access]
"""Provide optional Numba integration for map-folding state containers.

(AI generated docstring)

You can use this module to set Numba thread limits with the package `CPUlimit` convention and to
bridge `StateMapFolding` and `StateMapFoldingSymmetric` into Numba `StructRef` values [1][2][3]. The
module keeps the public Python state objects alive while Numba reads and reflects the compiled
payload fields.

Contents
--------
Functions
	defineProcessorLimitNumba
		Set Numba's worker-thread limit from the package CPU-limit convention.

Classes
	形NumbaMapFoldingState
		Represent a Numba `StructRef` view of map-folding state objects.

References
----------
[1] Numba documentation
	https://numba.readthedocs.io/en/stable/
[2] `mapFolding.dataBaskets.StateMapFolding`

[3] `mapFolding.dataBaskets.StateMapFoldingSymmetric`

"""
from __future__ import annotations

from contextlib import ExitStack
from hunterMakesPy.parseParameters import defineConcurrencyLimit
from mapFolding.dataBaskets import StateMapFolding, StateMapFoldingSymmetric
from numba import get_num_threads, set_num_threads, typeof
from numba.core import cgutils, types
from numba.experimental import structref
from numba.experimental.jitclass.base import imp_dtor
from numba.extending import box, NativeValue, reflect, typeof_impl, unbox
from typing import override, TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import Limitation
	from typing import Any

def defineProcessorLimitNumba(CPUlimit: Limitation) -> int:
	"""Set Numba's worker-thread limit from the package `CPUlimit` convention.

	(AI generated docstring)

	You can use this function to apply the package-wide `CPUlimit` convention to Numba's global
	thread setting [1][2]. `defineProcessorLimitNumba` reads the current Numba worker capacity,
	computes `concurrencyLimit` with `defineConcurrencyLimit`, updates Numba in place, and returns
	the effective thread count.

	Parameters
	----------
	CPUlimit : Limitation
		CPU-usage limit in the package convention defined by `defineConcurrencyLimit` [2].

	Returns
	-------
	concurrencyLimit : int
		Effective number of Numba worker threads after `set_num_threads` updates the process-global
		limit [1].

	Examples
	--------
	The repository dispatches to `defineProcessorLimitNumba` from
	`mapFolding.beDRY.defineProcessorLimit` [3].

	```python
	from mapFolding._optionalNumba import defineProcessorLimitNumba

	concurrencyLimit = defineProcessorLimitNumba(CPUlimit)
	```

	References
	----------
	[1] Numba documentation
		https://numba.readthedocs.io/en/stable/
	[2] hunterMakesPy - Context7
		https://context7.com/hunterhogan/huntermakespy
	[3] `mapFolding.beDRY.defineProcessorLimit`

	"""
	concurrencyLimit: int = defineConcurrencyLimit(limit=CPUlimit, cpuTotal=get_num_threads())
	set_num_threads(concurrencyLimit)
	return get_num_threads()

_numbaParentField: str = 'numbaParent'

@structref.register
class _形NumbaMapFoldingState(types.StructRef):
	@override
	def preprocess_fields(self, fields: tuple[tuple[str, types.Type], ...]) -> tuple[tuple[str, types.Type], ...]:
		def _unliteralField(field: tuple[str, types.Type]) -> tuple[str, types.Type]:
			return field[0], types.unliteral(field[1])

		return tuple(map(_unliteralField, fields))

形NumbaMapFoldingState = _形NumbaMapFoldingState

@typeof_impl.register(StateMapFoldingSymmetric)
@typeof_impl.register(StateMapFolding)
def _typeofMapFoldingState(value: StateMapFolding | StateMapFoldingSymmetric, _context: Any) -> _形NumbaMapFoldingState:
	def _typeofField(field: dataclasses.Field[Any]) -> tuple[str, types.Type]:
		return field.name, typeof(getattr(value, field.name))

	return _形NumbaMapFoldingState(((_numbaParentField, types.voidptr), *tuple(map(_typeofField, dataclasses.fields(value)))))

@unbox(_形NumbaMapFoldingState)
def _unboxMapFoldingState(numbaType: _形NumbaMapFoldingState, value: Any, context: Any) -> NativeValue:
	numbaContext = context.context
	payloadModel = numbaContext.data_model_manager[numbaType.get_data_type()]
	payloadLLVMtype = payloadModel.get_value_type()
	stateSlot = cgutils.alloca_once_value(context.builder, cgutils.get_null_value(numbaContext.get_value_type(numbaType)))
	memoryInformationSlot = cgutils.alloca_once_value(context.builder, cgutils.get_null_value(cgutils.voidptr_t))
	isErrorSlot = cgutils.alloca_once_value(context.builder, cgutils.false_bit)

	context.pyapi.incref(value)
	memoryInformation = numbaContext.nrt.meminfo_alloc_dtor_unchecked(
		context.builder
		, numbaContext.get_constant(types.uintp, numbaContext.get_abi_sizeof(payloadLLVMtype))
		, imp_dtor(numbaContext, context.builder.module, numbaType)
	)
	context.builder.store(memoryInformation, memoryInformationSlot)

	with context.builder.if_else(cgutils.is_null(context.builder, memoryInformation), likely=False) as (allocationFailed, allocationSucceeded):
		with allocationFailed:
			context.pyapi.err_set_none('PyExc_MemoryError')
			context.builder.store(cgutils.true_bit, isErrorSlot)
		with allocationSucceeded:
			payloadPointer = numbaContext.nrt.meminfo_data(context.builder, memoryInformation)
			payloadPointer = context.builder.bitcast(payloadPointer, payloadLLVMtype.as_pointer())
			context.builder.store(cgutils.get_null_value(payloadLLVMtype), payloadPointer)

			structReferenceUtilities = structref._Utils(numbaContext, context.builder, numbaType)
			state = structReferenceUtilities.new_struct_ref(memoryInformation)
			stateValue = state._getvalue()
			context.builder.store(stateValue, stateSlot)
			payload = structReferenceUtilities.get_data_struct(stateValue)
			setattr(payload, _numbaParentField, context.builder.bitcast(value, cgutils.voidptr_t))

			with ExitStack() as stack:
				def unboxField(fieldName: str) -> None:
					fieldType = numbaType.field_dict[fieldName]
					fieldObject = context.pyapi.object_getattr_string(value, fieldName)
					with cgutils.early_exit_if_null(context.builder, stack, fieldObject):
						context.builder.store(cgutils.true_bit, isErrorSlot)
					fieldNative = context.unbox(fieldType, fieldObject)
					context.pyapi.decref(fieldObject)
					with cgutils.early_exit_if(context.builder, stack, fieldNative.is_error):
						context.builder.store(cgutils.true_bit, isErrorSlot)
					setattr(payload, fieldName, fieldNative.value)

				tuple(map(unboxField, tuple(numbaType.field_dict)[1:]))

	stateValue = context.builder.load(stateSlot)
	isError = context.builder.load(isErrorSlot)
	with context.builder.if_then(isError, likely=False):
		context.pyapi.decref(value)
		with context.builder.if_then(cgutils.is_not_null(context.builder, context.builder.load(memoryInformationSlot))):
			numbaContext.nrt.decref(context.builder, numbaType, stateValue)

	def cleanUp() -> None:
		context.pyapi.decref(value)

	return NativeValue(stateValue, is_error=isError, cleanup=cleanUp)

@reflect(_形NumbaMapFoldingState)
def _reflectMapFoldingState(numbaType: _形NumbaMapFoldingState, value: Any, context: Any) -> None:
	payload = structref._Utils(context.context, context.builder, numbaType).get_data_struct(value)
	parent = context.builder.bitcast(getattr(payload, _numbaParentField), context.pyapi.pyobj)

	def reflectField(fieldName: str) -> None:
		fieldType = numbaType.field_dict[fieldName]
		fieldValue = getattr(payload, fieldName)
		context.context.nrt.incref(context.builder, fieldType, fieldValue)
		fieldObject = context.box(fieldType, fieldValue)
		context.pyapi.object_setattr_string(parent, fieldName, fieldObject)
		context.pyapi.decref(fieldObject)

	tuple(map(reflectField, tuple(numbaType.field_dict)[1:]))

@box(_形NumbaMapFoldingState)
def _boxMapFoldingState(numbaType: _形NumbaMapFoldingState, value: Any, context: Any) -> Any:
	payload = structref._Utils(context.context, context.builder, numbaType).get_data_struct(value)
	parent = context.builder.bitcast(getattr(payload, _numbaParentField), context.pyapi.pyobj)
	context.pyapi.incref(parent)
	context.context.nrt.decref(context.builder, numbaType, value)
	return parent
