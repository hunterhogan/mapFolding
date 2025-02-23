from _typeshed import Incomplete
from numba import prange as prange
from numba.core import config as config, errors as errors, types as types
from numba.core.extending import make_attribute_wrapper as make_attribute_wrapper, models as models, register_model as register_model, type_callable as type_callable, typeof_impl as typeof_impl
from numba.core.typing.templates import AbstractTemplate as AbstractTemplate, AttributeTemplate as AttributeTemplate, ConcreteTemplate as ConcreteTemplate, bound_function as bound_function, infer as infer, infer_getattr as infer_getattr, infer_global as infer_global, make_callable_template as make_callable_template, signature as signature
from numba.parfors.parfor import internal_prange as internal_prange

class Print(AbstractTemplate):
    def generic(self, args, kws): ...

class PrintItem(AbstractTemplate):
    key: str
    def generic(self, args, kws): ...

class Abs(ConcreteTemplate):
    int_cases: Incomplete
    real_cases: Incomplete
    complex_cases: Incomplete
    cases: Incomplete

class Slice(ConcreteTemplate):
    cases: Incomplete

class Range(ConcreteTemplate):
    cases: Incomplete

class GetIter(AbstractTemplate):
    key: str
    def generic(self, args, kws): ...

class IterNext(AbstractTemplate):
    key: str
    def generic(self, args, kws): ...

class PairFirst(AbstractTemplate):
    """
    Given a heterogeneous pair, return the first element.
    """
    key: str
    def generic(self, args, kws): ...

class PairSecond(AbstractTemplate):
    """
    Given a heterogeneous pair, return the second element.
    """
    key: str
    def generic(self, args, kws): ...

def choose_result_bitwidth(*inputs): ...
def choose_result_int(*inputs):
    """
    Choose the integer result type for an operation on integer inputs,
    according to the integer typing NBEP. In accordance with the new
    type system.
    """

all_ints: Incomplete
integer_binop_cases: Incomplete

class BinOp(ConcreteTemplate):
    cases: Incomplete

class BinOpAdd(BinOp): ...
class BinOpAdd(BinOp): ...
class BinOpSub(BinOp): ...
class BinOpSub(BinOp): ...
class BinOpMul(BinOp): ...
class BinOpMul(BinOp): ...

class BinOpMod(ConcreteTemplate):
    cases: Incomplete

class BinOpMod(ConcreteTemplate):
    cases: Incomplete

class BinOpTrueDiv(ConcreteTemplate):
    cases: Incomplete

class BinOpTrueDiv(ConcreteTemplate):
    cases: Incomplete

class BinOpFloorDiv(ConcreteTemplate):
    cases: Incomplete

class BinOpFloorDiv(ConcreteTemplate):
    cases: Incomplete

class DivMod(ConcreteTemplate):
    _tys: Incomplete
    cases: Incomplete

class BinOpPower(ConcreteTemplate):
    cases: Incomplete

class BinOpPower(ConcreteTemplate):
    cases: Incomplete

class PowerBuiltin(BinOpPower): ...

class BitwiseShiftOperation(ConcreteTemplate):
    cases: Incomplete
    unsafe_casting: bool

class BitwiseLeftShift(BitwiseShiftOperation): ...
class BitwiseLeftShift(BitwiseShiftOperation): ...
class BitwiseRightShift(BitwiseShiftOperation): ...
class BitwiseRightShift(BitwiseShiftOperation): ...

class BitwiseLogicOperation(BinOp):
    cases: Incomplete
    unsafe_casting: bool

class BitwiseAnd(BitwiseLogicOperation): ...
class BitwiseAnd(BitwiseLogicOperation): ...
class BitwiseOr(BitwiseLogicOperation): ...
class BitwiseOr(BitwiseLogicOperation): ...
class BitwiseXor(BitwiseLogicOperation): ...
class BitwiseXor(BitwiseLogicOperation): ...

class BitwiseInvert(ConcreteTemplate):
    cases: Incomplete
    unsafe_casting: bool

class UnaryOp(ConcreteTemplate):
    cases: Incomplete

class UnaryNegate(UnaryOp): ...
class UnaryPositive(UnaryOp): ...

class UnaryNot(ConcreteTemplate):
    cases: Incomplete

class OrderedCmpOp(ConcreteTemplate):
    cases: Incomplete

class UnorderedCmpOp(ConcreteTemplate):
    cases: Incomplete

class CmpOpLt(OrderedCmpOp): ...
class CmpOpLe(OrderedCmpOp): ...
class CmpOpGt(OrderedCmpOp): ...
class CmpOpGe(OrderedCmpOp): ...

class ConstOpEq(AbstractTemplate):
    def generic(self, args, kws): ...

class ConstOpNotEq(ConstOpEq): ...
class CmpOpEq(UnorderedCmpOp): ...
class CmpOpNe(UnorderedCmpOp): ...

class TupleCompare(AbstractTemplate):
    def generic(self, args, kws): ...

class TupleEq(TupleCompare): ...
class TupleNe(TupleCompare): ...
class TupleGe(TupleCompare): ...
class TupleGt(TupleCompare): ...
class TupleLe(TupleCompare): ...
class TupleLt(TupleCompare): ...

class TupleAdd(AbstractTemplate):
    def generic(self, args, kws): ...

class CmpOpIdentity(AbstractTemplate):
    def generic(self, args, kws): ...

class CmpOpIs(CmpOpIdentity): ...
class CmpOpIsNot(CmpOpIdentity): ...

def normalize_1d_index(index):
    """
    Normalize the *index* type (an integer or slice) for indexing a 1D
    sequence.
    """

class GetItemCPointer(AbstractTemplate):
    def generic(self, args, kws): ...

class SetItemCPointer(AbstractTemplate):
    def generic(self, args, kws): ...

class Len(AbstractTemplate):
    def generic(self, args, kws): ...

class TupleConstructor(AbstractTemplate):
    def generic(self, args, kws): ...

class Contains(AbstractTemplate):
    def generic(self, args, kws): ...

class TupleBool(AbstractTemplate):
    def generic(self, args, kws): ...

class StaticGetItemTuple(AbstractTemplate):
    key: str
    def generic(self, args, kws): ...

class StaticGetItemLiteralList(AbstractTemplate):
    key: str
    def generic(self, args, kws): ...

class StaticGetItemLiteralStrKeyDict(AbstractTemplate):
    key: str
    def generic(self, args, kws): ...

class StaticGetItemClass(AbstractTemplate):
    '''This handles the "static_getitem" when a Numba type is subscripted e.g:
    var = typed.List.empty_list(float64[::1, :])
    It only allows this on simple numerical types. Compound types, like
    records, are not supported.
    '''
    key: str
    def generic(self, args, kws): ...

class GenericNotIn(AbstractTemplate):
    key: str
    def generic(self, args, kws): ...

class MemoryViewAttribute(AttributeTemplate):
    key = types.MemoryView
    def resolve_contiguous(self, buf): ...
    def resolve_c_contiguous(self, buf): ...
    def resolve_f_contiguous(self, buf): ...
    def resolve_itemsize(self, buf): ...
    def resolve_nbytes(self, buf): ...
    def resolve_readonly(self, buf): ...
    def resolve_shape(self, buf): ...
    def resolve_strides(self, buf): ...
    def resolve_ndim(self, buf): ...

class BooleanAttribute(AttributeTemplate):
    key = types.Boolean
    def resolve___class__(self, ty): ...
    def resolve_item(self, ty, args, kws): ...

class NumberAttribute(AttributeTemplate):
    key = types.Number
    def resolve___class__(self, ty): ...
    def resolve_real(self, ty): ...
    def resolve_imag(self, ty): ...
    def resolve_conjugate(self, ty, args, kws): ...
    def resolve_item(self, ty, args, kws): ...

class NPTimedeltaAttribute(AttributeTemplate):
    key = types.NPTimedelta
    def resolve___class__(self, ty): ...

class NPDatetimeAttribute(AttributeTemplate):
    key = types.NPDatetime
    def resolve___class__(self, ty): ...

class SliceAttribute(AttributeTemplate):
    key = types.SliceType
    def resolve_start(self, ty): ...
    def resolve_stop(self, ty): ...
    def resolve_step(self, ty): ...
    def resolve_indices(self, ty, args, kws): ...

class NumberClassAttribute(AttributeTemplate):
    key = types.NumberClass
    def resolve___call__(self, classty):
        """
        Resolve a NumPy number class's constructor (e.g. calling numpy.int32(...))
        """

class TypeRefAttribute(AttributeTemplate):
    key = types.TypeRef
    context: Incomplete
    pysig: Incomplete
    def resolve___call__(self, classty):
        """
        Resolve a core number's constructor (e.g. calling int(...))

        Note:

        This is needed because of the limitation of the current type-system
        implementation.  Specifically, the lack of a higher-order type
        (i.e. passing the ``DictType`` vs ``DictType(key_type, value_type)``)
        """

class MinMaxBase(AbstractTemplate):
    def _unify_minmax(self, tys): ...
    def generic(self, args, kws):
        """
        Resolve a min() or max() call.
        """

class Max(MinMaxBase): ...
class Min(MinMaxBase): ...

class Round(ConcreteTemplate):
    cases: Incomplete

class Bool(AbstractTemplate):
    def generic(self, args, kws): ...

class Int(AbstractTemplate):
    def generic(self, args, kws): ...

class Float(AbstractTemplate):
    def generic(self, args, kws): ...

class Complex(AbstractTemplate):
    def generic(self, args, kws): ...

class Enumerate(AbstractTemplate):
    def generic(self, args, kws): ...

class Zip(AbstractTemplate):
    def generic(self, args, kws): ...

class Iter(AbstractTemplate):
    def generic(self, args, kws): ...

class Next(AbstractTemplate):
    def generic(self, args, kws): ...

class TypeBuiltin(AbstractTemplate):
    def generic(self, args, kws): ...

class OptionalAttribute(AttributeTemplate):
    key = types.Optional
    def generic_resolve(self, optional, attr): ...

class DeferredAttribute(AttributeTemplate):
    key = types.DeferredType
    def generic_resolve(self, deferred, attr): ...

class IndexValue:
    """
    Index and value
    """
    index: Incomplete
    value: Incomplete
    def __init__(self, ind, val) -> None: ...
    def __repr__(self) -> str: ...

class IndexValueType(types.Type):
    val_typ: Incomplete
    def __init__(self, val_typ) -> None: ...

def typeof_index(val, c): ...
def type_index_value(context): ...

class IndexValueModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None: ...
