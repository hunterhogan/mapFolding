from _typeshed import Incomplete
from numba.core import config as config, types as types

def jitclass(cls_or_spec: Incomplete | None = None, spec: Incomplete | None = None):
    '''
    A function for creating a jitclass.
    Can be used as a decorator or function.

    Different use cases will cause different arguments to be set.

    If specified, ``spec`` gives the types of class fields.
    It must be a dictionary or sequence.
    With a dictionary, use collections.OrderedDict for stable ordering.
    With a sequence, it must contain 2-tuples of (fieldname, fieldtype).

    Any class annotations for field names not listed in spec will be added.
    For class annotation `x: T` we will append ``("x", as_numba_type(T))`` to
    the spec if ``x`` is not already a key in spec.


    Examples
    --------

    1) ``cls_or_spec = None``, ``spec = None``

    >>> @jitclass()
    ... class Foo:
    ...     ...

    2) ``cls_or_spec = None``, ``spec = spec``

    >>> @jitclass(spec=spec)
    ... class Foo:
    ...     ...

    3) ``cls_or_spec = Foo``, ``spec = None``

    >>> @jitclass
    ... class Foo:
    ...     ...

    4) ``cls_or_spec = spec``, ``spec = None``
    In this case we update ``cls_or_spec, spec = None, cls_or_spec``.

    >>> @jitclass(spec)
    ... class Foo:
    ...     ...

    5) ``cls_or_spec = Foo``, ``spec = spec``

    >>> JitFoo = jitclass(Foo, spec)

    Returns
    -------
    If used as a decorator, returns a callable that takes a class object and
    returns a compiled version.
    If used as a function, returns the compiled class (an instance of
    ``JitClassType``).
    '''
