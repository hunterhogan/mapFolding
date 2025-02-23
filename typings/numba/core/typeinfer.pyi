from _typeshed import Incomplete
from collections.abc import Generator
from numba.core import config as config, ir as ir, types as types, typing as typing, utils as utils
from numba.core.errors import CompilerError as CompilerError, ForceLiteralArg as ForceLiteralArg, NumbaValueError as NumbaValueError, TypingError as TypingError, UnsupportedError as UnsupportedError, UntypedAttributeError as UntypedAttributeError, new_error_context as new_error_context, termcolor as termcolor
from numba.core.funcdesc import qualifying_prefix as qualifying_prefix
from numba.core.typeconv import Conversion as Conversion
from numba.core.typing.templates import Signature as Signature

_logger: Incomplete

class NOTSET: ...

_termcolor: Incomplete

class TypeVar:
    context: Incomplete
    var: Incomplete
    type: Incomplete
    locked: bool
    define_loc: Incomplete
    literal_value: Incomplete
    def __init__(self, context, var) -> None: ...
    def add_type(self, tp, loc): ...
    def lock(self, tp, loc, literal_value=...) -> None: ...
    def union(self, other, loc): ...
    def __repr__(self) -> str: ...
    @property
    def defined(self): ...
    def get(self): ...
    def getone(self): ...
    def __len__(self) -> int: ...

class ConstraintNetwork:
    """
    TODO: It is possible to optimize constraint propagation to consider only
          dirty type variables.
    """
    constraints: Incomplete
    def __init__(self) -> None: ...
    def append(self, constraint) -> None: ...
    def propagate(self, typeinfer):
        """
        Execute all constraints.  Errors are caught and returned as a list.
        This allows progressing even though some constraints may fail
        due to lack of information
        (e.g. imprecise types such as List(undefined)).
        """

class Propagate:
    """
    A simple constraint for direct propagation of types for assignments.
    """
    dst: Incomplete
    src: Incomplete
    loc: Incomplete
    def __init__(self, dst, src, loc) -> None: ...
    def __call__(self, typeinfer) -> None: ...
    def refine(self, typeinfer, target_type) -> None: ...

class ArgConstraint:
    dst: Incomplete
    src: Incomplete
    loc: Incomplete
    def __init__(self, dst, src, loc) -> None: ...
    def __call__(self, typeinfer) -> None: ...

class BuildTupleConstraint:
    target: Incomplete
    items: Incomplete
    loc: Incomplete
    def __init__(self, target, items, loc) -> None: ...
    def __call__(self, typeinfer) -> None: ...

class _BuildContainerConstraint:
    target: Incomplete
    items: Incomplete
    loc: Incomplete
    def __init__(self, target, items, loc) -> None: ...
    def __call__(self, typeinfer) -> None: ...

class BuildListConstraint(_BuildContainerConstraint):
    target: Incomplete
    items: Incomplete
    loc: Incomplete
    def __init__(self, target, items, loc) -> None: ...
    def __call__(self, typeinfer) -> None: ...

class BuildSetConstraint(_BuildContainerConstraint):
    container_type = types.Set

class BuildMapConstraint:
    target: Incomplete
    items: Incomplete
    special_value: Incomplete
    value_indexes: Incomplete
    loc: Incomplete
    def __init__(self, target, items, special_value, value_indexes, loc) -> None: ...
    def __call__(self, typeinfer): ...

class ExhaustIterConstraint:
    target: Incomplete
    count: Incomplete
    iterator: Incomplete
    loc: Incomplete
    def __init__(self, target, count, iterator, loc) -> None: ...
    def __call__(self, typeinfer) -> None: ...

class PairFirstConstraint:
    target: Incomplete
    pair: Incomplete
    loc: Incomplete
    def __init__(self, target, pair, loc) -> None: ...
    def __call__(self, typeinfer) -> None: ...

class PairSecondConstraint:
    target: Incomplete
    pair: Incomplete
    loc: Incomplete
    def __init__(self, target, pair, loc) -> None: ...
    def __call__(self, typeinfer) -> None: ...

class StaticGetItemConstraint:
    target: Incomplete
    value: Incomplete
    index: Incomplete
    fallback: Incomplete
    loc: Incomplete
    def __init__(self, target, value, index, index_var, loc) -> None: ...
    def __call__(self, typeinfer) -> None: ...
    def get_call_signature(self): ...

class TypedGetItemConstraint:
    target: Incomplete
    value: Incomplete
    dtype: Incomplete
    index: Incomplete
    loc: Incomplete
    def __init__(self, target, value, dtype, index, loc) -> None: ...
    signature: Incomplete
    def __call__(self, typeinfer) -> None: ...
    def get_call_signature(self): ...

def fold_arg_vars(typevars, args, vararg, kws):
    """
    Fold and resolve the argument variables of a function call.
    """
def _is_array_not_precise(arrty):
    """Check type is array and it is not precise
    """

class CallConstraint:
    """Constraint for calling functions.
    Perform case analysis foreach combinations of argument types.
    """
    signature: Incomplete
    target: Incomplete
    func: Incomplete
    args: Incomplete
    kws: Incomplete
    vararg: Incomplete
    loc: Incomplete
    def __init__(self, target, func, args, kws, vararg, loc) -> None: ...
    def __call__(self, typeinfer) -> None: ...
    def resolve(self, typeinfer, typevars, fnty) -> None: ...
    def _add_refine_map(self, typeinfer, typevars, sig) -> None:
        """Add this expression to the refine_map base on the type of target_type
        """
    def refine(self, typeinfer, updated_type) -> None: ...
    def get_call_signature(self): ...

class IntrinsicCallConstraint(CallConstraint):
    def __call__(self, typeinfer) -> None: ...

class GetAttrConstraint:
    target: Incomplete
    attr: Incomplete
    value: Incomplete
    loc: Incomplete
    inst: Incomplete
    def __init__(self, target, attr, value, loc, inst) -> None: ...
    def __call__(self, typeinfer) -> None: ...
    def refine(self, typeinfer, target_type) -> None: ...
    def __repr__(self) -> str: ...

class SetItemRefinement:
    """A mixin class to provide the common refinement logic in setitem
    and static setitem.
    """
    def _refine_target_type(self, typeinfer, targetty, idxty, valty, sig) -> None:
        """Refine the target-type given the known index type and value type.
        """

class SetItemConstraint(SetItemRefinement):
    target: Incomplete
    index: Incomplete
    value: Incomplete
    loc: Incomplete
    def __init__(self, target, index, value, loc) -> None: ...
    signature: Incomplete
    def __call__(self, typeinfer) -> None: ...
    def get_call_signature(self): ...

class StaticSetItemConstraint(SetItemRefinement):
    target: Incomplete
    index: Incomplete
    index_var: Incomplete
    value: Incomplete
    loc: Incomplete
    def __init__(self, target, index, index_var, value, loc) -> None: ...
    signature: Incomplete
    def __call__(self, typeinfer) -> None: ...
    def get_call_signature(self): ...

class DelItemConstraint:
    target: Incomplete
    index: Incomplete
    loc: Incomplete
    def __init__(self, target, index, loc) -> None: ...
    signature: Incomplete
    def __call__(self, typeinfer) -> None: ...
    def get_call_signature(self): ...

class SetAttrConstraint:
    target: Incomplete
    attr: Incomplete
    value: Incomplete
    loc: Incomplete
    def __init__(self, target, attr, value, loc) -> None: ...
    signature: Incomplete
    def __call__(self, typeinfer) -> None: ...
    def get_call_signature(self): ...

class PrintConstraint:
    args: Incomplete
    vararg: Incomplete
    loc: Incomplete
    def __init__(self, args, vararg, loc) -> None: ...
    signature: Incomplete
    def __call__(self, typeinfer) -> None: ...
    def get_call_signature(self): ...

class TypeVarMap(dict):
    context: Incomplete
    def set_context(self, context) -> None: ...
    def __getitem__(self, name): ...
    def __setitem__(self, name, value) -> None: ...

_temporary_dispatcher_map: Incomplete
_temporary_dispatcher_map_ref_count: Incomplete

def register_dispatcher(disp) -> Generator[None]:
    """
    Register a Dispatcher for inference while it is not yet stored
    as global or closure variable (e.g. during execution of the @jit()
    call).  This allows resolution of recursive calls with eager
    compilation.
    """

typeinfer_extensions: Incomplete

class TypeInferer:
    """
    Operates on block that shares the same ir.Scope.
    """
    context: Incomplete
    blocks: Incomplete
    generator_info: Incomplete
    func_id: Incomplete
    func_ir: Incomplete
    typevars: Incomplete
    constraints: Incomplete
    warnings: Incomplete
    arg_names: Incomplete
    assumed_immutables: Incomplete
    calls: Incomplete
    calltypes: Incomplete
    refine_map: Incomplete
    debug: Incomplete
    _skip_recursion: bool
    def __init__(self, context, func_ir, warnings) -> None: ...
    def copy(self, skip_recursion: bool = False): ...
    def _mangle_arg_name(self, name): ...
    def _get_return_vars(self): ...
    def get_argument_types(self): ...
    def seed_argument(self, name, index, typ) -> None: ...
    def seed_type(self, name, typ) -> None:
        """All arguments should be seeded.
        """
    def seed_return(self, typ) -> None:
        """Seeding of return value is optional.
        """
    def build_constraint(self) -> None: ...
    def return_types_from_partial(self):
        """
        Resume type inference partially to deduce the return type.
        Note: No side-effect to `self`.

        Returns the inferred return type or None if it cannot deduce the return
        type.
        """
    def propagate(self, raise_errors: bool = True): ...
    def add_type(self, var, tp, loc, unless_locked: bool = False) -> None: ...
    def add_calltype(self, inst, signature) -> None: ...
    def copy_type(self, src_var, dest_var, loc) -> None: ...
    def lock_type(self, var, tp, loc, literal_value=...) -> None: ...
    def propagate_refined_type(self, updated_var, updated_type) -> None: ...
    def unify(self, raise_errors: bool = True):
        """
        Run the final unification pass over all inferred types, and
        catch imprecise types.
        """
    def get_generator_type(self, typdict, retty, raise_errors: bool = True): ...
    def get_function_types(self, typemap):
        """
        Fill and return the calltypes map.
        """
    def _unify_return_types(self, rettypes): ...
    def get_return_type(self, typemap): ...
    def get_state_token(self):
        '''The algorithm is monotonic.  It can only grow or "refine" the
        typevar map.
        '''
    def constrain_statement(self, inst) -> None: ...
    def typeof_setitem(self, inst) -> None: ...
    def typeof_storemap(self, inst) -> None: ...
    def typeof_static_setitem(self, inst) -> None: ...
    def typeof_delitem(self, inst) -> None: ...
    def typeof_setattr(self, inst) -> None: ...
    def typeof_print(self, inst) -> None: ...
    def typeof_assign(self, inst) -> None: ...
    def resolve_value_type(self, inst, val):
        """
        Resolve the type of a simple Python value, such as can be
        represented by literals.
        """
    def typeof_arg(self, inst, target, arg) -> None: ...
    def typeof_const(self, inst, target, const) -> None: ...
    def typeof_yield(self, inst, target, yield_) -> None: ...
    def sentry_modified_builtin(self, inst, gvar) -> None:
        """
        Ensure that builtins are not modified.
        """
    def resolve_call(self, fnty, pos_args, kw_args):
        """
        Resolve a call to a given function type.  A signature is returned.
        """
    def typeof_global(self, inst, target, gvar): ...
    def typeof_expr(self, inst, target, expr) -> None: ...
    def typeof_call(self, inst, target, call) -> None: ...
    def typeof_intrinsic_call(self, inst, target, func, *args) -> None: ...

class NullDebug:
    def propagate_started(self) -> None: ...
    def propagate_finished(self) -> None: ...
    def unify_finished(self, typdict, retty, fntys) -> None: ...

class TypeInferDebug:
    typeinfer: Incomplete
    def __init__(self, typeinfer) -> None: ...
    def _dump_state(self) -> None: ...
    def propagate_started(self) -> None: ...
    def propagate_finished(self) -> None: ...
    def unify_finished(self, typdict, retty, fntys) -> None: ...
