from _typeshed import Incomplete
from collections.abc import Generator
from llvmlite.ir import instructions as instructions, types as types, values as values

_CMP_MAP: Incomplete

def _unop(opname, cls=...): ...
def _binop(opname, cls=...): ...
def _binop_with_overflow(opname, cls=...): ...
def _uniop(opname, cls=...): ...
def _uniop_intrinsic_int(opname): ...
def _uniop_intrinsic_float(opname): ...
def _uniop_intrinsic_with_flag(opname): ...
def _triop_intrinsic(opname): ...
def _castop(opname, cls=...): ...
def _label_suffix(label, suffix):
    """Returns (label + suffix) or a truncated version if it's too long.
    Parameters
    ----------
    label : str
        Label name
    suffix : str
        Label suffix
    """

class IRBuilder:
    _block: Incomplete
    _anchor: Incomplete
    debug_metadata: Incomplete
    def __init__(self, block: Incomplete | None = None) -> None: ...
    @property
    def block(self):
        """
        The current basic block.
        """
    basic_block = block
    @property
    def function(self):
        """
        The current function.
        """
    @property
    def module(self):
        """
        The current module.
        """
    def position_before(self, instr) -> None:
        """
        Position immediately before the given instruction.  The current block
        is also changed to the instruction's basic block.
        """
    def position_after(self, instr) -> None:
        """
        Position immediately after the given instruction.  The current block
        is also changed to the instruction's basic block.
        """
    def position_at_start(self, block) -> None:
        """
        Position at the start of the basic *block*.
        """
    def position_at_end(self, block) -> None:
        """
        Position at the end of the basic *block*.
        """
    def append_basic_block(self, name: str = ''):
        """
        Append a basic block, with the given optional *name*, to the current
        function.  The current block is not changed.  The new block is returned.
        """
    def remove(self, instr) -> None:
        """Remove the given instruction."""
    def goto_block(self, block) -> Generator[None]:
        """
        A context manager which temporarily positions the builder at the end
        of basic block *bb* (but before any terminator).
        """
    def goto_entry_block(self) -> Generator[None]:
        """
        A context manager which temporarily positions the builder at the
        end of the function's entry block.
        """
    def _branch_helper(self, bbenter, bbexit) -> Generator[Incomplete]: ...
    def if_then(self, pred, likely: Incomplete | None = None) -> Generator[Incomplete]:
        """
        A context manager which sets up a conditional basic block based
        on the given predicate (a i1 value).  If the conditional block
        is not explicitly terminated, a branch will be added to the next
        block.
        If *likely* is given, its boolean value indicates whether the
        predicate is likely to be true or not, and metadata is issued
        for LLVM's optimizers to account for that.
        """
    def if_else(self, pred, likely: Incomplete | None = None) -> Generator[Incomplete]:
        """
        A context manager which sets up two conditional basic blocks based
        on the given predicate (a i1 value).
        A tuple of context managers is yield'ed.  Each context manager
        acts as a if_then() block.
        *likely* has the same meaning as in if_then().

        Typical use::
            with builder.if_else(pred) as (then, otherwise):
                with then:
                    # emit instructions for when the predicate is true
                with otherwise:
                    # emit instructions for when the predicate is false
        """
    def _insert(self, instr) -> None: ...
    def _set_terminator(self, term): ...
    def shl(self, lhs, rhs, name: str = '') -> None:
        """
        Left integer shift:
            name = lhs << rhs
        """
    def lshr(self, lhs, rhs, name: str = '') -> None:
        """
        Logical (unsigned) right integer shift:
            name = lhs >> rhs
        """
    def ashr(self, lhs, rhs, name: str = '') -> None:
        """
        Arithmetic (signed) right integer shift:
            name = lhs >> rhs
        """
    def add(self, lhs, rhs, name: str = '') -> None:
        """
        Integer addition:
            name = lhs + rhs
        """
    def fadd(self, lhs, rhs, name: str = '') -> None:
        """
        Floating-point addition:
            name = lhs + rhs
        """
    def sub(self, lhs, rhs, name: str = '') -> None:
        """
        Integer subtraction:
            name = lhs - rhs
        """
    def fsub(self, lhs, rhs, name: str = '') -> None:
        """
        Floating-point subtraction:
            name = lhs - rhs
        """
    def mul(self, lhs, rhs, name: str = '') -> None:
        """
        Integer multiplication:
            name = lhs * rhs
        """
    def fmul(self, lhs, rhs, name: str = '') -> None:
        """
        Floating-point multiplication:
            name = lhs * rhs
        """
    def udiv(self, lhs, rhs, name: str = '') -> None:
        """
        Unsigned integer division:
            name = lhs / rhs
        """
    def sdiv(self, lhs, rhs, name: str = '') -> None:
        """
        Signed integer division:
            name = lhs / rhs
        """
    def fdiv(self, lhs, rhs, name: str = '') -> None:
        """
        Floating-point division:
            name = lhs / rhs
        """
    def urem(self, lhs, rhs, name: str = '') -> None:
        """
        Unsigned integer remainder:
            name = lhs % rhs
        """
    def srem(self, lhs, rhs, name: str = '') -> None:
        """
        Signed integer remainder:
            name = lhs % rhs
        """
    def frem(self, lhs, rhs, name: str = '') -> None:
        """
        Floating-point remainder:
            name = lhs % rhs
        """
    def or_(self, lhs, rhs, name: str = '') -> None:
        """
        Bitwise integer OR:
            name = lhs | rhs
        """
    def and_(self, lhs, rhs, name: str = '') -> None:
        """
        Bitwise integer AND:
            name = lhs & rhs
        """
    def xor(self, lhs, rhs, name: str = '') -> None:
        """
        Bitwise integer XOR:
            name = lhs ^ rhs
        """
    def sadd_with_overflow(self, lhs, rhs, name: str = '') -> None:
        """
        Signed integer addition with overflow:
            name = {result, overflow bit} = lhs + rhs
        """
    def smul_with_overflow(self, lhs, rhs, name: str = '') -> None:
        """
        Signed integer multiplication with overflow:
            name = {result, overflow bit} = lhs * rhs
        """
    def ssub_with_overflow(self, lhs, rhs, name: str = '') -> None:
        """
        Signed integer subtraction with overflow:
            name = {result, overflow bit} = lhs - rhs
        """
    def uadd_with_overflow(self, lhs, rhs, name: str = '') -> None:
        """
        Unsigned integer addition with overflow:
            name = {result, overflow bit} = lhs + rhs
        """
    def umul_with_overflow(self, lhs, rhs, name: str = '') -> None:
        """
        Unsigned integer multiplication with overflow:
            name = {result, overflow bit} = lhs * rhs
        """
    def usub_with_overflow(self, lhs, rhs, name: str = '') -> None:
        """
        Unsigned integer subtraction with overflow:
            name = {result, overflow bit} = lhs - rhs
        """
    def not_(self, value, name: str = ''):
        """
        Bitwise integer complement:
            name = ~value
        """
    def neg(self, value, name: str = ''):
        """
        Integer negative:
            name = -value
        """
    def fneg(self, arg, name: str = '', flags=()) -> None:
        """
        Floating-point negative:
            name = -arg
        """
    def _icmp(self, prefix, cmpop, lhs, rhs, name): ...
    def icmp_signed(self, cmpop, lhs, rhs, name: str = ''):
        """
        Signed integer comparison:
            name = lhs <cmpop> rhs

        where cmpop can be '==', '!=', '<', '<=', '>', '>='
        """
    def icmp_unsigned(self, cmpop, lhs, rhs, name: str = ''):
        """
        Unsigned integer (or pointer) comparison:
            name = lhs <cmpop> rhs

        where cmpop can be '==', '!=', '<', '<=', '>', '>='
        """
    def fcmp_ordered(self, cmpop, lhs, rhs, name: str = '', flags=()):
        """
        Floating-point ordered comparison:
            name = lhs <cmpop> rhs

        where cmpop can be '==', '!=', '<', '<=', '>', '>=', 'ord', 'uno'
        """
    def fcmp_unordered(self, cmpop, lhs, rhs, name: str = '', flags=()):
        """
        Floating-point unordered comparison:
            name = lhs <cmpop> rhs

        where cmpop can be '==', '!=', '<', '<=', '>', '>=', 'ord', 'uno'
        """
    def select(self, cond, lhs, rhs, name: str = '', flags=()):
        """
        Ternary select operator:
            name = cond ? lhs : rhs
        """
    def trunc(self, value, typ, name: str = '') -> None:
        """
        Truncating integer downcast to a smaller type:
            name = (typ) value
        """
    def zext(self, value, typ, name: str = '') -> None:
        """
        Zero-extending integer upcast to a larger type:
            name = (typ) value
        """
    def sext(self, value, typ, name: str = '') -> None:
        """
        Sign-extending integer upcast to a larger type:
            name = (typ) value
        """
    def fptrunc(self, value, typ, name: str = '') -> None:
        """
        Floating-point downcast to a less precise type:
            name = (typ) value
        """
    def fpext(self, value, typ, name: str = '') -> None:
        """
        Floating-point upcast to a more precise type:
            name = (typ) value
        """
    def bitcast(self, value, typ, name: str = '') -> None:
        """
        Pointer cast to a different pointer type:
            name = (typ) value
        """
    def addrspacecast(self, value, typ, name: str = '') -> None:
        """
        Pointer cast to a different address space:
            name = (typ) value
        """
    def fptoui(self, value, typ, name: str = '') -> None:
        """
        Convert floating-point to unsigned integer:
            name = (typ) value
        """
    def uitofp(self, value, typ, name: str = '') -> None:
        """
        Convert unsigned integer to floating-point:
            name = (typ) value
        """
    def fptosi(self, value, typ, name: str = '') -> None:
        """
        Convert floating-point to signed integer:
            name = (typ) value
        """
    def sitofp(self, value, typ, name: str = '') -> None:
        """
        Convert signed integer to floating-point:
            name = (typ) value
        """
    def ptrtoint(self, value, typ, name: str = '') -> None:
        """
        Cast pointer to integer:
            name = (typ) value
        """
    def inttoptr(self, value, typ, name: str = '') -> None:
        """
        Cast integer to pointer:
            name = (typ) value
        """
    def alloca(self, typ, size: Incomplete | None = None, name: str = ''):
        """
        Stack-allocate a slot for *size* elements of the given type.
        (default one element)
        """
    def load(self, ptr, name: str = '', align: Incomplete | None = None, typ: Incomplete | None = None):
        """
        Load value from pointer, with optional guaranteed alignment:
            name = *ptr
        """
    def store(self, value, ptr, align: Incomplete | None = None):
        """
        Store value to pointer, with optional guaranteed alignment:
            *ptr = name
        """
    def load_atomic(self, ptr, ordering, align, name: str = '', typ: Incomplete | None = None):
        """
        Load value from pointer, with optional guaranteed alignment:
            name = *ptr
        """
    def store_atomic(self, value, ptr, ordering, align):
        """
        Store value to pointer, with optional guaranteed alignment:
            *ptr = name
        """
    def switch(self, value, default):
        """
        Create a switch-case with a single *default* target.
        """
    def branch(self, target):
        """
        Unconditional branch to *target*.
        """
    def cbranch(self, cond, truebr, falsebr):
        """
        Conditional branch to *truebr* if *cond* is true, else to *falsebr*.
        """
    def branch_indirect(self, addr):
        """
        Indirect branch to target *addr*.
        """
    def ret_void(self):
        """
        Return from function without a value.
        """
    def ret(self, value):
        """
        Return from function with the given *value*.
        """
    def resume(self, landingpad):
        """
        Resume an in-flight exception.
        """
    def call(self, fn, args, name: str = '', cconv: Incomplete | None = None, tail: bool = False, fastmath=(), attrs=(), arg_attrs: Incomplete | None = None):
        """
        Call function *fn* with *args*:
            name = fn(args...)
        """
    def asm(self, ftype, asm, constraint, args, side_effect, name: str = ''):
        """
        Inline assembler.
        """
    def load_reg(self, reg_type, reg_name, name: str = ''):
        '''
        Load a register value into an LLVM value.
          Example: v = load_reg(IntType(32), "eax")
        '''
    def store_reg(self, value, reg_type, reg_name, name: str = ''):
        '''
        Store an LLVM value inside a register
        Example:
          store_reg(Constant(IntType(32), 0xAAAAAAAA), IntType(32), "eax")
        '''
    def invoke(self, fn, args, normal_to, unwind_to, name: str = '', cconv: Incomplete | None = None, fastmath=(), attrs=(), arg_attrs: Incomplete | None = None): ...
    def gep(self, ptr, indices, inbounds: bool = False, name: str = '', source_etype: Incomplete | None = None):
        """
        Compute effective address (getelementptr):
            name = getelementptr ptr, <indices...>
        """
    def extract_element(self, vector, idx, name: str = ''):
        """
        Returns the value at position idx.
        """
    def insert_element(self, vector, value, idx, name: str = ''):
        """
        Returns vector with vector[idx] replaced by value.
        The result is undefined if the idx is larger or equal the vector length.
        """
    def shuffle_vector(self, vector1, vector2, mask, name: str = ''):
        """
        Constructs a permutation of elements from *vector1* and *vector2*.
        Returns a new vector in the same length of *mask*.

        * *vector1* and *vector2* must have the same element type.
        * *mask* must be a constant vector of integer types.
        """
    def extract_value(self, agg, idx, name: str = ''):
        """
        Extract member number *idx* from aggregate.
        """
    def insert_value(self, agg, value, idx, name: str = ''):
        """
        Insert *value* into member number *idx* from aggregate.
        """
    def phi(self, typ, name: str = '', flags=()): ...
    def unreachable(self): ...
    def atomic_rmw(self, op, ptr, val, ordering, name: str = ''): ...
    def cmpxchg(self, ptr, cmp, val, ordering, failordering: Incomplete | None = None, name: str = ''):
        """
        Atomic compared-and-set:
            atomic {
                old = *ptr
                success = (old == cmp)
                if (success)
                    *ptr = val
                }
            name = { old, success }

        If failordering is `None`, the value of `ordering` is used.
        """
    def landingpad(self, typ, name: str = '', cleanup: bool = False): ...
    def assume(self, cond):
        """
        Optimizer hint: assume *cond* is always true.
        """
    def fence(self, ordering, targetscope: Incomplete | None = None, name: str = ''):
        """
        Add a memory barrier, preventing certain reorderings of load and/or
        store accesses with
        respect to other processors and devices.
        """
    def comment(self, text):
        """
        Puts a single-line comment into the generated IR. This will be ignored
        by LLVM, but can be useful for debugging the output of a compiler. Adds
        a comment to the source file.

        * *text* is a string that does not contain new line characters.
        """
    def bswap(self, cond) -> None:
        """
        Used to byte swap integer values with an even number of bytes (positive
        multiple of 16 bits)
        """
    def bitreverse(self, cond) -> None:
        """
        Reverse the bitpattern of an integer value; for example 0b10110110
        becomes 0b01101101.
        """
    def ctpop(self, cond) -> None:
        """
        Counts the number of bits set in a value.
        """
    def ctlz(self, cond, flag) -> None:
        """
        Counts leading zero bits in *value*. Boolean *flag* indicates whether
        the result is defined for ``0``.
        """
    def cttz(self, cond, flag) -> None:
        """
        Counts trailing zero bits in *value*. Boolean *flag* indicates whether
        the result is defined for ``0``.
        """
    def fma(self, a, b, c) -> None:
        """
        Perform the fused multiply-add operation.
        """
    def convert_from_fp16(self, a, to: Incomplete | None = None, name: str = ''):
        """
        Convert from an i16 to the given FP type
        """
    def convert_to_fp16(self, a) -> None:
        """
        Convert the given FP number to an i16
        """
