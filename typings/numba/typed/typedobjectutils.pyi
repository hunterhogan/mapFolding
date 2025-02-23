from numba.core import cgutils as cgutils, types as types, typing as typing
from numba.core.errors import NumbaTypeError as NumbaTypeError, NumbaTypeSafetyWarning as NumbaTypeSafetyWarning, TypingError as TypingError
from numba.core.extending import intrinsic as intrinsic
from numba.core.registry import cpu_target as cpu_target
from numba.core.typeconv import Conversion as Conversion

def _as_bytes(builder, ptr):
    """Helper to do (void*)ptr
    """
def _cast(typingctx, val, typ):
    """Cast *val* to *typ*
    """
def _sentry_safe_cast(fromty, toty):
    """Check and raise TypingError if *fromty* cannot be safely cast to *toty*
    """
def _sentry_safe_cast_default(default, valty):
    """Similar to _sentry_safe_cast but handle default value.
    """
def _nonoptional(typingctx, val):
    """Typing trick to cast Optional[T] to T
    """
def _container_get_data(context, builder, container_ty, c):
    """Helper to get the C list pointer in a numba containers.
    """
def _container_get_meminfo(context, builder, container_ty, c):
    """Helper to get the meminfo for a container
    """
def _get_incref_decref(context, module, datamodel, container_element_type): ...
def _get_equal(context, module, datamodel, container_element_type): ...
