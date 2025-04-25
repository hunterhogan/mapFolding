"""This file is generated automatically, so changes to this file will be lost."""
from mapFolding import astDOTParamSpec, astDOTTryStar, astDOTTypeAlias, astDOTTypeVar, astDOTTypeVarTuple, astDOTtype_param
from mapFolding.someAssemblyRequired._astTypes import *
from typing import Any, overload
import ast
'# ruff: noqa: F405'

class DOT:
    """
	Access attributes and sub-nodes of AST elements via consistent accessor methods.

	The DOT class provides static methods to access specific attributes of different types of AST nodes in a consistent
	way. This simplifies attribute access across various node types and improves code readability by abstracting the
	underlying AST structure details.

	DOT is designed for safe, read-only access to node properties, unlike the grab class which is designed for modifying
	node attributes.
	"""

    @staticmethod
    @overload
    def annotation(node: hasDOTannotation_expr):
        ...

    @staticmethod
    @overload
    def annotation(node: hasDOTannotation_exprOrNone):
        ...

    @staticmethod
    def annotation(node: hasDOTannotation):
        return node.annotation

    @staticmethod
    @overload
    def arg(node: hasDOTarg_Identifier):
        ...

    @staticmethod
    @overload
    def arg(node: hasDOTarg_IdentifierOrNone):
        ...

    @staticmethod
    def arg(node: hasDOTarg):
        return node.arg

    @staticmethod
    @overload
    def args(node: hasDOTargs_list_arg):
        ...

    @staticmethod
    @overload
    def args(node: hasDOTargs_arguments):
        ...

    @staticmethod
    @overload
    def args(node: hasDOTargs_list_expr):
        ...

    @staticmethod
    def args(node: hasDOTargs):
        return node.args

    @staticmethod
    def argtypes(node: hasDOTargtypes):
        return node.argtypes

    @staticmethod
    def asname(node: hasDOTasname):
        return node.asname

    @staticmethod
    def attr(node: hasDOTattr):
        return node.attr

    @staticmethod
    def bases(node: hasDOTbases):
        return node.bases

    @staticmethod
    @overload
    def body(node: hasDOTbody_list_stmt):
        ...

    @staticmethod
    @overload
    def body(node: hasDOTbody_expr):
        ...

    @staticmethod
    def body(node: hasDOTbody):
        return node.body

    @staticmethod
    def bound(node: hasDOTbound):
        return node.bound

    @staticmethod
    def cases(node: hasDOTcases):
        return node.cases

    @staticmethod
    def cause(node: hasDOTcause):
        return node.cause

    @staticmethod
    def cls(node: hasDOTcls):
        return node.cls

    @staticmethod
    def comparators(node: hasDOTcomparators):
        return node.comparators

    @staticmethod
    def context_expr(node: hasDOTcontext_expr):
        return node.context_expr

    @staticmethod
    def conversion(node: hasDOTconversion):
        return node.conversion

    @staticmethod
    def ctx(node: hasDOTctx):
        return node.ctx

    @staticmethod
    def decorator_list(node: hasDOTdecorator_list):
        return node.decorator_list

    @staticmethod
    def default_value(node: hasDOTdefault_value):
        return node.default_value

    @staticmethod
    def defaults(node: hasDOTdefaults):
        return node.defaults

    @staticmethod
    def elt(node: hasDOTelt):
        return node.elt

    @staticmethod
    def elts(node: hasDOTelts):
        return node.elts

    @staticmethod
    def exc(node: hasDOTexc):
        return node.exc

    @staticmethod
    def finalbody(node: hasDOTfinalbody):
        return node.finalbody

    @staticmethod
    def format_spec(node: hasDOTformat_spec):
        return node.format_spec

    @staticmethod
    def func(node: hasDOTfunc):
        return node.func

    @staticmethod
    def generators(node: hasDOTgenerators):
        return node.generators

    @staticmethod
    def guard(node: hasDOTguard):
        return node.guard

    @staticmethod
    def handlers(node: hasDOThandlers):
        return node.handlers

    @staticmethod
    def id(node: hasDOTid):
        return node.id

    @staticmethod
    def ifs(node: hasDOTifs):
        return node.ifs

    @staticmethod
    def is_async(node: hasDOTis_async):
        return node.is_async

    @staticmethod
    def items(node: hasDOTitems):
        return node.items

    @staticmethod
    def iter(node: hasDOTiter):
        return node.iter

    @staticmethod
    def key(node: hasDOTkey):
        return node.key

    @staticmethod
    @overload
    def keys(node: hasDOTkeys_list_exprOrNone):
        ...

    @staticmethod
    @overload
    def keys(node: hasDOTkeys_list_expr):
        ...

    @staticmethod
    def keys(node: hasDOTkeys):
        return node.keys

    @staticmethod
    def keywords(node: hasDOTkeywords):
        return node.keywords

    @staticmethod
    def kind(node: hasDOTkind):
        return node.kind

    @staticmethod
    def kw_defaults(node: hasDOTkw_defaults):
        return node.kw_defaults

    @staticmethod
    def kwarg(node: hasDOTkwarg):
        return node.kwarg

    @staticmethod
    def kwd_attrs(node: hasDOTkwd_attrs):
        return node.kwd_attrs

    @staticmethod
    def kwd_patterns(node: hasDOTkwd_patterns):
        return node.kwd_patterns

    @staticmethod
    def kwonlyargs(node: hasDOTkwonlyargs):
        return node.kwonlyargs

    @staticmethod
    def left(node: hasDOTleft):
        return node.left

    @staticmethod
    def level(node: hasDOTlevel):
        return node.level

    @staticmethod
    def lineno(node: hasDOTlineno):
        return node.lineno

    @staticmethod
    def lower(node: hasDOTlower):
        return node.lower

    @staticmethod
    def module(node: hasDOTmodule):
        return node.module

    @staticmethod
    def msg(node: hasDOTmsg):
        return node.msg

    @staticmethod
    @overload
    def name(node: hasDOTname_Identifier):
        ...

    @staticmethod
    @overload
    def name(node: hasDOTname_IdentifierOrNone):
        ...

    @staticmethod
    @overload
    def name(node: hasDOTname_Name):
        ...

    @staticmethod
    def name(node: hasDOTname):
        return node.name

    @staticmethod
    @overload
    def names(node: hasDOTnames_list_Identifier):
        ...

    @staticmethod
    @overload
    def names(node: hasDOTnames_list_alias):
        ...

    @staticmethod
    def names(node: hasDOTnames):
        return node.names

    @staticmethod
    @overload
    def op(node: hasDOTop_operator):
        ...

    @staticmethod
    @overload
    def op(node: hasDOTop_boolop):
        ...

    @staticmethod
    @overload
    def op(node: hasDOTop_unaryop):
        ...

    @staticmethod
    def op(node: hasDOTop):
        return node.op

    @staticmethod
    def operand(node: hasDOToperand):
        return node.operand

    @staticmethod
    def ops(node: hasDOTops):
        return node.ops

    @staticmethod
    def optional_vars(node: hasDOToptional_vars):
        return node.optional_vars

    @staticmethod
    @overload
    def orelse(node: hasDOTorelse_list_stmt):
        ...

    @staticmethod
    @overload
    def orelse(node: hasDOTorelse_expr):
        ...

    @staticmethod
    def orelse(node: hasDOTorelse):
        return node.orelse

    @staticmethod
    @overload
    def pattern(node: hasDOTpattern_pattern):
        ...

    @staticmethod
    @overload
    def pattern(node: hasDOTpattern_patternOrNone):
        ...

    @staticmethod
    def pattern(node: hasDOTpattern):
        return node.pattern

    @staticmethod
    def patterns(node: hasDOTpatterns):
        return node.patterns

    @staticmethod
    def posonlyargs(node: hasDOTposonlyargs):
        return node.posonlyargs

    @staticmethod
    def rest(node: hasDOTrest):
        return node.rest

    @staticmethod
    @overload
    def returns(node: hasDOTreturns_exprOrNone):
        ...

    @staticmethod
    @overload
    def returns(node: hasDOTreturns_expr):
        ...

    @staticmethod
    def returns(node: hasDOTreturns):
        return node.returns

    @staticmethod
    def right(node: hasDOTright):
        return node.right

    @staticmethod
    def simple(node: hasDOTsimple):
        return node.simple

    @staticmethod
    def slice(node: hasDOTslice):
        return node.slice

    @staticmethod
    def step(node: hasDOTstep):
        return node.step

    @staticmethod
    def subject(node: hasDOTsubject):
        return node.subject

    @staticmethod
    def tag(node: hasDOTtag):
        return node.tag

    @staticmethod
    @overload
    def target(node: hasDOTtarget_NameOrAttributeOrSubscript):
        ...

    @staticmethod
    @overload
    def target(node: hasDOTtarget_expr):
        ...

    @staticmethod
    @overload
    def target(node: hasDOTtarget_Name):
        ...

    @staticmethod
    def target(node: hasDOTtarget):
        return node.target

    @staticmethod
    def targets(node: hasDOTtargets):
        return node.targets

    @staticmethod
    def test(node: hasDOTtest):
        return node.test

    @staticmethod
    def type(node: hasDOTtype):
        return node.type

    @staticmethod
    def type_comment(node: hasDOTtype_comment):
        return node.type_comment

    @staticmethod
    def type_ignores(node: hasDOTtype_ignores):
        return node.type_ignores

    @staticmethod
    def type_params(node: hasDOTtype_params):
        return node.type_params

    @staticmethod
    def upper(node: hasDOTupper):
        return node.upper

    @staticmethod
    @overload
    def value(node: hasDOTvalue_exprOrNone):
        ...

    @staticmethod
    @overload
    def value(node: hasDOTvalue_expr):
        ...

    @staticmethod
    @overload
    def value(node: hasDOTvalue_Any):
        ...

    @staticmethod
    @overload
    def value(node: hasDOTvalue_LiteralTrueFalseOrNone):
        ...

    @staticmethod
    def value(node: hasDOTvalue):
        return node.value

    @staticmethod
    def values(node: hasDOTvalues):
        return node.values

    @staticmethod
    def vararg(node: hasDOTvararg):
        return node.vararg