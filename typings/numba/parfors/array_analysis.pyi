from _typeshed import Incomplete
from numba.core import cgutils as cgutils, config as config, errors as errors, ir as ir, types as types
from numba.core.analysis import compute_cfg_from_blocks as compute_cfg_from_blocks
from numba.core.extending import intrinsic as intrinsic
from numba.core.ir_utils import GuardException as GuardException, build_definitions as build_definitions, dprint_func_ir as dprint_func_ir, find_build_sequence as find_build_sequence, find_callname as find_callname, find_const as find_const, find_potential_aliases as find_potential_aliases, find_topo_order as find_topo_order, get_canonical_alias as get_canonical_alias, get_definition as get_definition, get_global_func_typ as get_global_func_typ, guard as guard, is_namedtuple_class as is_namedtuple_class, mk_unique_var as mk_unique_var, require as require
from numba.core.typing import npydecl as npydecl, signature as signature

UNKNOWN_CLASS: int
CONST_CLASS: int
MAP_TYPES: Incomplete
array_analysis_extensions: Incomplete
array_creation: Incomplete
random_int_args: Incomplete
random_1arg_size: Incomplete
random_2arg_sizelast: Incomplete
random_3arg_sizelast: Incomplete
random_calls: Incomplete

def wrap_index(typingctx, idx, size):
    '''
    Calculate index value "idx" relative to a size "size" value as
    (idx % size), where "size" is known to be positive.
    Note that we use the mod(%) operation here instead of
    (idx < 0 ? idx + size : idx) because we may have situations
    where idx > size due to the way indices are calculated
    during slice/range analysis.

    Both idx and size have to be Integer types.
    size should be from the array size vars that array_analysis
    adds and the bitwidth should match the platform maximum.
    '''
def wrap_index_literal(idx, size): ...
def assert_equiv(typingctx, *val):
    """
    A function that asserts the inputs are of equivalent size,
    and throws runtime error when they are not. The input is
    a vararg that contains an error message, followed by a set
    of objects of either array, tuple or integer.
    """

class EquivSet:
    """EquivSet keeps track of equivalence relations between
    a set of objects.
    """
    obj_to_ind: Incomplete
    ind_to_obj: Incomplete
    next_ind: Incomplete
    def __init__(self, obj_to_ind: Incomplete | None = None, ind_to_obj: Incomplete | None = None, next_ind: int = 0) -> None:
        """Create a new EquivSet object. Optional keyword arguments are for
        internal use only.
        """
    def empty(self):
        """Return an empty EquivSet object.
        """
    def clone(self):
        """Return a new copy.
        """
    def __repr__(self) -> str: ...
    def is_empty(self):
        """Return true if the set is empty, or false otherwise.
        """
    def _get_ind(self, x):
        """Return the internal index (greater or equal to 0) of the given
        object, or -1 if not found.
        """
    def _get_or_add_ind(self, x):
        """Return the internal index (greater or equal to 0) of the given
        object, or create a new one if not found.
        """
    def _insert(self, objs) -> None:
        """Base method that inserts a set of equivalent objects by modifying
        self.
        """
    def is_equiv(self, *objs):
        """Try to derive if given objects are equivalent, return true
        if so, or false otherwise.
        """
    def get_equiv_const(self, obj):
        """Check if obj is equivalent to some int constant, and return
        the constant if found, or None otherwise.
        """
    def get_equiv_set(self, obj):
        """Return the set of equivalent objects.
        """
    def insert_equiv(self, *objs):
        """Insert a set of equivalent objects by modifying self. This
        method can be overloaded to transform object type before insertion.
        """
    def intersect(self, equiv_set):
        """ Return the intersection of self and the given equiv_set,
        without modifying either of them. The result will also keep
        old equivalence indices unchanged.
        """

class ShapeEquivSet(EquivSet):
    """Just like EquivSet, except that it accepts only numba IR variables
    and constants as objects, guided by their types. Arrays are considered
    equivalent as long as their shapes are equivalent. Scalars are
    equivalent only when they are equal in value. Tuples are equivalent
    when they are of the same size, and their elements are equivalent.
    """
    typemap: Incomplete
    defs: Incomplete
    ind_to_var: Incomplete
    ind_to_const: Incomplete
    def __init__(self, typemap, defs: Incomplete | None = None, ind_to_var: Incomplete | None = None, obj_to_ind: Incomplete | None = None, ind_to_obj: Incomplete | None = None, next_id: int = 0, ind_to_const: Incomplete | None = None) -> None:
        """Create a new ShapeEquivSet object, where typemap is a dictionary
        that maps variable names to their types, and it will not be modified.
        Optional keyword arguments are for internal use only.
        """
    def empty(self):
        """Return an empty ShapeEquivSet.
        """
    def clone(self):
        """Return a new copy.
        """
    def __repr__(self) -> str: ...
    def _get_names(self, obj):
        """Return a set of names for the given obj, where array and tuples
        are broken down to their individual shapes or elements. This is
        safe because both Numba array shapes and Python tuples are immutable.
        """
    def is_equiv(self, *objs):
        """Overload EquivSet.is_equiv to handle Numba IR variables and
        constants.
        """
    def get_equiv_const(self, obj):
        """If the given object is equivalent to a constant scalar,
        return the scalar value, or None otherwise.
        """
    def get_equiv_var(self, obj):
        """If the given object is equivalent to some defined variable,
        return the variable, or None otherwise.
        """
    def get_equiv_set(self, obj):
        """Return the set of equivalent objects.
        """
    def _insert(self, objs) -> None:
        """Overload EquivSet._insert to manage ind_to_var dictionary.
        """
    def insert_equiv(self, *objs):
        """Overload EquivSet.insert_equiv to handle Numba IR variables and
        constants. Input objs are either variable or constant, and at least
        one of them must be variable.
        """
    def has_shape(self, name):
        """Return true if the shape of the given variable is available.
        """
    def get_shape(self, name):
        """Return a tuple of variables that corresponds to the shape
        of the given array, or None if not found.
        """
    def _get_shape(self, name):
        """Return a tuple of variables that corresponds to the shape
        of the given array, or raise GuardException if not found.
        """
    def get_shape_classes(self, name):
        """Instead of the shape tuple, return tuple of int, where
        each int is the corresponding class index of the size object.
        Unknown shapes are given class index -1. Return empty tuple
        if the input name is a scalar variable.
        """
    def intersect(self, equiv_set):
        """Overload the intersect method to handle ind_to_var.
        """
    def define(self, name, redefined) -> None:
        """Increment the internal count of how many times a variable is being
        defined. Most variables in Numba IR are SSA, i.e., defined only once,
        but not all of them. When a variable is being re-defined, it must
        be removed from the equivalence relation and added to the redefined
        set but only if that redefinition is not known to have the same
        equivalence classes. Those variables redefined are removed from all
        the blocks' equivalence sets later.

        Arrays passed to define() use their whole name but these do not
        appear in the equivalence sets since they are stored there per
        dimension. Calling _get_names() here converts array names to
        dimensional names.

        This function would previously invalidate if there were any multiple
        definitions of a variable.  However, we realized that this behavior
        is overly restrictive.  You need only invalidate on multiple
        definitions if they are not known to be equivalent. So, the
        equivalence insertion functions now return True if some change was
        made (meaning the definition was not equivalent) and False
        otherwise. If no change was made, then define() need not be
        called. For no change to have been made, the variable must
        already be present. If the new definition of the var has the
        case where lhs and rhs are in the same equivalence class then
        again, no change will be made and define() need not be called
        or the variable invalidated.
        """
    def union_defs(self, defs, redefined) -> None:
        """Union with the given defs dictionary. This is meant to handle
        branch join-point, where a variable may have been defined in more
        than one branches.
        """

class SymbolicEquivSet(ShapeEquivSet):
    """Just like ShapeEquivSet, except that it also reasons about variable
    equivalence symbolically by using their arithmetic definitions.
    The goal is to automatically derive the equivalence of array ranges
    (slicing). For instance, a[1:m] and a[0:m-1] shall be considered
    size-equivalence.
    """
    def_by: Incomplete
    ref_by: Incomplete
    ext_shapes: Incomplete
    rel_map: Incomplete
    wrap_map: Incomplete
    def __init__(self, typemap, def_by: Incomplete | None = None, ref_by: Incomplete | None = None, ext_shapes: Incomplete | None = None, defs: Incomplete | None = None, ind_to_var: Incomplete | None = None, obj_to_ind: Incomplete | None = None, ind_to_obj: Incomplete | None = None, next_id: int = 0) -> None:
        """Create a new SymbolicEquivSet object, where typemap is a dictionary
        that maps variable names to their types, and it will not be modified.
        Optional keyword arguments are for internal use only.
        """
    def empty(self):
        """Return an empty SymbolicEquivSet.
        """
    def __repr__(self) -> str: ...
    def clone(self):
        """Return a new copy.
        """
    def get_rel(self, name):
        """Retrieve a definition pair for the given variable,
        or return None if it is not available.
        """
    def _get_or_set_rel(self, name, func_ir: Incomplete | None = None):
        """Retrieve a definition pair for the given variable,
        and if it is not already available, try to look it up
        in the given func_ir, and remember it for future use.
        """
    def define(self, var, redefined, func_ir: Incomplete | None = None, typ: Incomplete | None = None):
        """Besides incrementing the definition count of the given variable
        name, it will also retrieve and simplify its definition from func_ir,
        and remember the result for later equivalence comparison. Supported
        operations are:
          1. arithmetic plus and minus with constants
          2. wrap_index (relative to some given size)
        """
    def _insert(self, objs):
        """Overload _insert method to handle ind changes between relative
        objects.  Returns True if some change is made, false otherwise.
        """
    def set_shape_setitem(self, obj, shape) -> None:
        """remember shapes of SetItem IR nodes.
        """
    def _get_shape(self, obj):
        """Overload _get_shape to retrieve the shape of SetItem IR nodes.
        """

class WrapIndexMeta:
    """
      Array analysis should be able to analyze all the function
      calls that it adds to the IR.  That way, array analysis can
      be run as often as needed and you should get the same
      equivalencies.  One modification to the IR that array analysis
      makes is the insertion of wrap_index calls.  Thus, repeated
      array analysis passes should be able to analyze these wrap_index
      calls.  The difficulty of these calls is that the equivalence
      class of the left-hand side of the assignment is not present in
      the arguments to wrap_index in the right-hand side.  Instead,
      the equivalence class of the wrap_index output is a combination
      of the wrap_index args.  The important thing to
      note is that if the equivalence classes of the slice size
      and the dimension's size are the same for two wrap index
      calls then we can be assured of the answer being the same.
      So, we maintain the wrap_map dict that maps from a tuple
      of equivalence class ids for the slice and dimension size
      to some new equivalence class id for the output size.
      However, when we are analyzing the first such wrap_index
      call we don't have a variable there to associate to the
      size since we're in the process of analyzing the instruction
      that creates that mapping.  So, instead we return an object
      of this special class and analyze_inst will establish the
      connection between a tuple of the parts of this object
      below and the left-hand side variable.
    """
    slice_size: Incomplete
    dim_size: Incomplete
    def __init__(self, slice_size, dim_size) -> None: ...

class ArrayAnalysis:
    aa_count: int
    context: Incomplete
    func_ir: Incomplete
    typemap: Incomplete
    calltypes: Incomplete
    equiv_sets: Incomplete
    array_attr_calls: Incomplete
    object_attrs: Incomplete
    prepends: Incomplete
    pruned_predecessors: Incomplete
    def __init__(self, context, func_ir, typemap, calltypes) -> None: ...
    def get_equiv_set(self, block_label):
        """Return the equiv_set object of an block given its label.
        """
    def remove_redefineds(self, redefineds) -> None:
        """Take a set of variables in redefineds and go through all
        the currently existing equivalence sets (created in topo order)
        and remove that variable from all of them since it is multiply
        defined within the function.
        """
    def run(self, blocks: Incomplete | None = None, equiv_set: Incomplete | None = None) -> None:
        """run array shape analysis on the given IR blocks, resulting in
        modified IR and finalized EquivSet for each block.
        """
    def _run_on_blocks(self, topo_order, blocks, cfg, init_equiv_set) -> None: ...
    def _combine_to_new_block(self, block, pending_transforms) -> None:
        """Combine the new instructions from previous pass into a new block
        body.
        """
    def _determine_transform(self, cfg, block, label, scope, init_equiv_set):
        """Determine the transformation for each instruction in the block
        """
    def dump(self) -> None:
        """dump per-block equivalence sets for debugging purposes.
        """
    def _define(self, equiv_set, var, typ, value) -> None: ...
    class AnalyzeResult:
        kwargs: Incomplete
        def __init__(self, **kwargs) -> None: ...
    def _analyze_inst(self, label, scope, equiv_set, inst, redefined): ...
    def _analyze_expr(self, scope, equiv_set, expr, lhs): ...
    def _analyze_op_getattr(self, scope, equiv_set, expr, lhs): ...
    def _analyze_op_cast(self, scope, equiv_set, expr, lhs): ...
    def _analyze_op_exhaust_iter(self, scope, equiv_set, expr, lhs): ...
    def gen_literal_slice_part(self, arg_val, loc, scope, stmts, equiv_set, name: str = 'static_literal_slice_part'): ...
    def gen_static_slice_size(self, lhs_rel, rhs_rel, loc, scope, stmts, equiv_set): ...
    def gen_explicit_neg(self, arg, arg_rel, arg_typ, size_typ, loc, scope, dsize, stmts, equiv_set): ...
    def update_replacement_slice(self, lhs, lhs_typ, lhs_rel, dsize_rel, replacement_slice, slice_index, need_replacement, loc, scope, stmts, equiv_set, size_typ, dsize): ...
    def slice_size(self, index, dsize, equiv_set, scope, stmts):
        '''Reason about the size of a slice represented by the "index"
        variable, and return a variable that has this size data, or
        raise GuardException if it cannot reason about it.

        The computation takes care of negative values used in the slice
        with respect to the given dimensional size ("dsize").

        Extra statements required to produce the result are appended
        to parent function\'s stmts list.
        '''
    def _index_to_shape(self, scope, equiv_set, var, ind_var):
        """For indexing like var[index] (either write or read), see if
        the index corresponds to a range/slice shape.
        Returns a 2-tuple where the first item is either None or a ir.Var
        to be used to replace the index variable in the outer getitem or
        setitem instruction.  The second item is also a tuple returning
        the shape and prepending instructions.
        """
    def _analyze_op_getitem(self, scope, equiv_set, expr, lhs): ...
    def _analyze_op_static_getitem(self, scope, equiv_set, expr, lhs): ...
    def _analyze_op_unary(self, scope, equiv_set, expr, lhs): ...
    def _analyze_op_binop(self, scope, equiv_set, expr, lhs): ...
    def _analyze_op_inplace_binop(self, scope, equiv_set, expr, lhs): ...
    def _analyze_op_arrayexpr(self, scope, equiv_set, expr, lhs): ...
    def _analyze_op_build_tuple(self, scope, equiv_set, expr, lhs): ...
    def _analyze_op_call(self, scope, equiv_set, expr, lhs): ...
    def _analyze_op_call_builtins_len(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numba_parfors_array_analysis_assert_equiv(self, scope, equiv_set, loc, args, kws) -> None: ...
    def _analyze_op_call_numba_parfors_array_analysis_wrap_index(self, scope, equiv_set, loc, args, kws):
        """ Analyze wrap_index calls added by a previous run of
            Array Analysis
        """
    def _analyze_numpy_create_array(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_empty(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numba_np_unsafe_ndarray_empty_inferred(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_zeros(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_ones(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_eye(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_identity(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_diag(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_numpy_array_like(self, scope, equiv_set, args, kws): ...
    def _analyze_op_call_numpy_ravel(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_copy(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_empty_like(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_zeros_like(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_ones_like(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_full_like(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_asfortranarray(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_reshape(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_transpose(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_rand(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_randn(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_numpy_random_with_size(self, pos, scope, equiv_set, args, kws): ...
    def _analyze_op_call_numpy_random_ranf(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_random_sample(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_sample(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_random(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_standard_normal(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_chisquare(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_weibull(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_power(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_geometric(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_exponential(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_poisson(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_rayleigh(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_normal(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_uniform(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_beta(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_binomial(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_f(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_gamma(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_lognormal(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_laplace(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_randint(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_random_triangular(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_concatenate(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_stack(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_vstack(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_hstack(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_dstack(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_cumsum(self, scope, equiv_set, loc, args, kws) -> None: ...
    def _analyze_op_call_numpy_cumprod(self, scope, equiv_set, loc, args, kws) -> None: ...
    def _analyze_op_call_numpy_linspace(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_op_call_numpy_dot(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_stencil(self, scope, equiv_set, stencil_func, loc, args, kws): ...
    def _analyze_op_call_numpy_linalg_inv(self, scope, equiv_set, loc, args, kws): ...
    def _analyze_broadcast(self, scope, equiv_set, loc, args, fn):
        """Infer shape equivalence of arguments based on Numpy broadcast rules
        and return shape of output
        https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        """
    def _broadcast_assert_shapes(self, scope, equiv_set, loc, shapes, names):
        """Produce assert_equiv for sizes in each dimension, taking into
        account of dimension coercion and constant size of 1.
        """
    def _call_assert_equiv(self, scope, loc, equiv_set, args, names: Incomplete | None = None): ...
    def _make_assert_equiv(self, scope, loc, equiv_set, _args, names: Incomplete | None = None): ...
    def _gen_shape_call(self, equiv_set, var, ndims, shape, post): ...
    def _isarray(self, varname): ...
    def _istuple(self, varname): ...
    def _sum_size(self, equiv_set, sizes):
        """Return the sum of the given list of sizes if they are all equivalent
        to some constant, or None otherwise.
        """

UNARY_MAP_OP: Incomplete
BINARY_MAP_OP: Incomplete
INPLACE_BINARY_MAP_OP: Incomplete
UFUNC_MAP_OP: Incomplete
