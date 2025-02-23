from _typeshed import Incomplete
from numba.core import ir as ir, types as types
from numba.core.typing import signature as signature
from typing import NamedTuple

class _CallableNode(NamedTuple):
    func: Incomplete
    sig: Incomplete

class ParforLoweringBuilder:
    """Helper class for building Numba-IR and lowering for Parfor.
    """
    _lowerer: Incomplete
    _scope: Incomplete
    _loc: Incomplete
    def __init__(self, lowerer, scope, loc) -> None: ...
    @property
    def _context(self): ...
    @property
    def _typingctx(self): ...
    @property
    def _typemap(self): ...
    @property
    def _calltypes(self): ...
    def bind_global_function(self, fobj, ftype, args, kws={}):
        """Binds a global function to a variable.

        Parameters
        ----------
        fobj : object
            The function to be bound.
        ftype : types.Type
        args : Sequence[types.Type]
        kws : Mapping[str, types.Type]

        Returns
        -------
        callable: _CallableNode
        """
    def make_const_variable(self, cval, typ, name: str = 'pf_const') -> ir.Var:
        """Makes a constant variable

        Parameters
        ----------
        cval : object
            The constant value
        typ : types.Type
            type of the value
        name : str
            variable name to store to

        Returns
        -------
        res : ir.Var
        """
    def make_tuple_variable(self, varlist, name: str = 'pf_tuple') -> ir.Var:
        """Makes a tuple variable

        Parameters
        ----------
        varlist : Sequence[ir.Var]
            Variables containing the values to be stored.
        name : str
            variable name to store to

        Returns
        -------
        res : ir.Var
        """
    def assign(self, rhs, typ, name: str = 'pf_assign') -> ir.Var:
        """Assign a value to a new variable

        Parameters
        ----------
        rhs : object
            The value
        typ : types.Type
            type of the value
        name : str
            variable name to store to

        Returns
        -------
        res : ir.Var
        """
    def assign_inplace(self, rhs, typ, name) -> ir.Var:
        """Assign a value to a new variable or inplace if it already exist

        Parameters
        ----------
        rhs : object
            The value
        typ : types.Type
            type of the value
        name : str
            variable name to store to

        Returns
        -------
        res : ir.Var
        """
    def call(self, callable_node, args, kws={}) -> ir.Expr:
        """Call a bound callable

        Parameters
        ----------
        callable_node : _CallableNode
            The callee
        args : Sequence[ir.Var]
        kws : Mapping[str, ir.Var]

        Returns
        -------
        res : ir.Expr
            The expression node for the return value of the call
        """
    def setitem(self, obj, index, val) -> ir.SetItem:
        """Makes a setitem call

        Parameters
        ----------
        obj : ir.Var
            the object being indexed
        index : ir.Var
            the index
        val : ir.Var
            the value to be stored

        Returns
        -------
        res : ir.SetItem
        """
    def getitem(self, obj, index, typ) -> ir.Expr:
        """Makes a getitem call

        Parameters
        ----------
        obj : ir.Var
            the object being indexed
        index : ir.Var
            the index
        val : ir.Var
            the ty

        Returns
        -------
        res : ir.Expr
            the retrieved value
        """
