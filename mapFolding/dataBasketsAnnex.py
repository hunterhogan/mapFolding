# ruff: ignore[undocumented-public-module]
# DEVELOPMENT
# ruff: file-ignore[private-member-access]
from __future__ import annotations

from contextlib import ExitStack
from mapFolding.dataBaskets import StateMapFolding, StateMapFoldingSymmetric
from numba import typeof, types as numba_types  # pyright: ignore[reportUnknownVariableType]
from numba.core import cgutils
from numba.experimental import structref
from numba.experimental.jitclass.base import imp_dtor  # pyright: ignore[reportUnknownVariableType]
from numba.extending import box, NativeValue, reflect, typeof_impl, unbox  # pyright: ignore[reportUnknownVariableType]
from typing import cast, NamedTuple, override, TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
	from collections.abc import Sequence
	from llvmlite.ir import AllocaInstr, IRBuilder, Type as ir_Type
	from numba.core.base import BaseContext
	from numba.core.datamodel.models import DataModel
	from numba.core.pythonapi import EnvironmentManager, PythonAPI
	from typing import Any

class _BoxContext(NamedTuple):
	context: BaseContext
	builder: IRBuilder
	pyapi: PythonAPI
	env_manager: EnvironmentManager

# box attribute
class _ReflectContext(NamedTuple):
	context: BaseContext
	builder: IRBuilder
	pyapi: PythonAPI
	env_manager: EnvironmentManager
	is_error: AllocaInstr

# unbox attribute
class _UnboxContext(NamedTuple):
	context: BaseContext
	builder: IRBuilder
	pyapi: PythonAPI

@structref.register  # pyright: ignore[reportUnknownMemberType]
class 形DataclassNumba(numba_types.StructRef):
	"""Represent Python dataclass structures in Numba-compiled code.

	(AI generated docstring)

	You can use this type to bridge Python dataclasses to Numba's type system. This
	`StructRef` [1] subclass enables Numba-compiled functions to manipulate dataclass
	instances by converting Python dataclass fields to Numba struct fields. The type
	preprocessing converts Numba literal types [2] to their unliteral equivalents,
	which allows Numba to generate efficient compiled code for dataclass operations.

	Numba Behavior
	--------------
	Numba requires type preprocessing for `StructRef` subclasses because Numba's type
	inference can produce literal types that must be normalized before compilation.
	This class applies `numba.types.unliteral` [3] to each field type during
	preprocessing. The unliteral transformation converts compile-time constant types
	to their runtime equivalents.

	See Also
	--------
	`_typeofDataclass`
		Register `StateMapFolding` and `StateMapFoldingSymmetric` to this type.

	References
	----------
	[1] `numba.experimental.structref.StructRef`
		https://numba.readthedocs.io/en/stable/reference/pysupported.html#structref

	[2] Numba literal types - Numba documentation
		https://numba.readthedocs.io/en/stable/reference/types.html#literal-types

	[3] `numba.types.unliteral`
		https://numba.readthedocs.io/en/stable/reference/types.html
	"""

	@override
	def preprocess_fields(self, fields: Sequence[tuple[str, numba_types.Type]]) -> tuple[tuple[str, numba_types.Type], ...]:
		def unliteral(aField: tuple[str, numba_types.Type]) -> tuple[str, numba_types.Type]:
			return (aField[0], numba_types.unliteral(aField[1]))
		return tuple(map(unliteral, fields))

形StateMapFoldingNumba = 形DataclassNumba

@typeof_impl.register(StateMapFoldingSymmetric)
@typeof_impl.register(StateMapFolding)
def _typeofDataclass(value: StateMapFolding | StateMapFoldingSymmetric, _context: Any) -> 形DataclassNumba:
	def typeofField(field: dataclasses.Field[Any]) -> tuple[str, numba_types.Type]:
		return field.name, typeof(getattr(value, field.name))
	return 形DataclassNumba((('fieldNumba', numba_types.voidptr), *tuple(map(typeofField, dataclasses.fields(value)))))

@unbox(形DataclassNumba)
def _unbox(numbaType: 形DataclassNumba, value: Any, context: _UnboxContext) -> NativeValue:
	numbaContext: BaseContext = context.context
	Z0Z_irType: ir_Type = cast('DataModel', numbaContext.data_model_manager[numbaType.get_data_type()]).get_value_type()
	stateSlot: AllocaInstr = cgutils.alloca_once_value(context.builder, cgutils.get_null_value(numbaContext.get_value_type(numbaType)))
	memoryInformationSlot: AllocaInstr = cgutils.alloca_once_value(context.builder, cgutils.get_null_value(cgutils.voidptr_t))
	isErrorSlot: AllocaInstr = cgutils.alloca_once_value(context.builder, cgutils.false_bit)

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
	context.pyapi.incref(value)
	memoryInformation = numbaContext.nrt.meminfo_alloc_dtor_unchecked(
		context.builder
		, numbaContext.get_constant(numba_types.uintp, numbaContext.get_abi_sizeof(Z0Z_irType))
		, imp_dtor(numbaContext, context.builder.module, numbaType)
	)
	context.builder.store(memoryInformation, memoryInformationSlot)

	with context.builder.if_else(cgutils.is_null(context.builder, memoryInformation), likely=False) as (allocationFailed, allocationSucceeded):  # pyright: ignore[reportGeneralTypeIssues] # ty: ignore[invalid-context-manager]
		with allocationFailed:
			context.pyapi.err_set_none('PyExc_MemoryError')
			context.builder.store(cgutils.true_bit, isErrorSlot)
		with allocationSucceeded:
			payloadPointer = numbaContext.nrt.meminfo_data(context.builder, memoryInformation)
			payloadPointer = context.builder.bitcast(payloadPointer, Z0Z_irType.as_pointer())
			context.builder.store(cgutils.get_null_value(Z0Z_irType), payloadPointer)

			structReferenceUtilities = structref._Utils(numbaContext, context.builder, numbaType)
			state = structReferenceUtilities.new_struct_ref(memoryInformation)
			stateValue = state._getvalue()
			context.builder.store(stateValue, stateSlot)
			payload = structReferenceUtilities.get_data_struct(stateValue)
			payload.fieldNumba = context.builder.bitcast(value, cgutils.voidptr_t)

			with ExitStack() as stack:
				def unboxField(fieldName: str) -> None:
					fieldType = numbaType.field_dict[fieldName]
					fieldObject = context.pyapi.object_getattr_string(value, fieldName)
					with cgutils.early_exit_if_null(context.builder, stack, fieldObject):
						context.builder.store(cgutils.true_bit, isErrorSlot)
					fieldNative = context.unbox(fieldType, fieldObject)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
					context.pyapi.decref(fieldObject)
					with cgutils.early_exit_if(context.builder, stack, fieldNative.is_error):
						context.builder.store(cgutils.true_bit, isErrorSlot)
					setattr(payload, fieldName, fieldNative.value)

				tuple(map(unboxField, tuple(numbaType.field_dict)[1:]))

	stateValue = context.builder.load(stateSlot)
	isError = context.builder.load(isErrorSlot)
	with context.builder.if_then(isError, likely=False):  # pyright: ignore[reportGeneralTypeIssues]  # ty: ignore[invalid-context-manager]
		context.pyapi.decref(value)
		with context.builder.if_then(cgutils.is_not_null(context.builder, context.builder.load(memoryInformationSlot))):  # pyright: ignore[reportGeneralTypeIssues]  # ty: ignore[invalid-context-manager]
			numbaContext.nrt.decref(context.builder, numbaType, stateValue)

	def cleanUp() -> None:
		context.pyapi.decref(value)

	return NativeValue(stateValue, is_error=isError, cleanup=cleanUp)

@reflect(形DataclassNumba)
def _reflect(numbaType: 形DataclassNumba, value: Any, context: _ReflectContext) -> None:
	payload = structref._Utils(context.context, context.builder, numbaType).get_data_struct(value)
	parent = context.builder.bitcast(payload.fieldNumba, context.pyapi.pyobj)

	def reflectField(fieldName: str) -> None:
		fieldType: numba_types.Type = numbaType.field_dict[fieldName]
		fieldValue = getattr(payload, fieldName)
		context.context.nrt.incref(context.builder, fieldType, fieldValue)
		fieldObject = context.box(fieldType, fieldValue)  # pyright: ignore[reportAttributeAccessIssue] # ty: ignore[unresolved-attribute]
		context.pyapi.object_setattr_string(parent, fieldName, fieldObject)
		context.pyapi.decref(fieldObject)

	tuple(map(reflectField, tuple(numbaType.field_dict)[1:]))

@box(形DataclassNumba)
def _box(numbaType: 形DataclassNumba, value: Any, context: _BoxContext) -> Any:
	payload = structref._Utils(context.context, context.builder, numbaType).get_data_struct(value)
	parent = context.builder.bitcast(payload.fieldNumba, context.pyapi.pyobj)
	context.pyapi.incref(parent)
	context.context.nrt.decref(context.builder, numbaType, value)
	return parent
