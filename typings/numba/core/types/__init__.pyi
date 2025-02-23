from .abstract import *
from .containers import *
from .functions import *
from .iterators import *
from .misc import *
from .npytypes import *
from .scalars import *
from .function_type import *
from .new_scalars import *
from _typeshed import Incomplete

__all__ = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'intp', 'uintp', 'intc', 'uintc', 'ssize_t', 'size_t', 'boolean', 'float32', 'float64', 'complex64', 'complex128', 'bool_', 'byte', 'char', 'uchar', 'short', 'ushort', 'int_', 'uint', 'long_', 'ulong', 'longlong', 'ulonglong', 'double', 'void', 'none', 'b1', 'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8', 'c8', 'c16', 'optional', 'ffi_forced_object', 'ffi', 'deferred_type', 'bool']

ffi_forced_object: Incomplete
ffi: Incomplete
none: Incomplete
string = unicode_type
optional = Optional
deferred_type = DeferredType
void = none
boolean: Incomplete
bool_: Incomplete
bool = bool_
byte: Incomplete
uint8: Incomplete
uint16: Incomplete
uint32: Incomplete
uint64: Incomplete
int8: Incomplete
int16: Incomplete
int32: Incomplete
int64: Incomplete
intp: Incomplete
uintp: Incomplete
intc: Incomplete
uintc: Incomplete
ssize_t: Incomplete
size_t: Incomplete
float32: Incomplete
float64: Incomplete
complex64: Incomplete
complex128: Incomplete
integer_domain = signed_domain | unsigned_domain
number_domain = real_domain | integer_domain | complex_domain
c_bool = boolean
py_bool = boolean
np_bool_ = boolean
c_uint8 = uint8
np_uint8 = uint8
c_uint16 = uint16
np_uint16 = uint16
c_uint32 = uint32
np_uint32 = uint32
c_uint64 = uint64
np_uint64 = uint64
c_uintp = uintp
np_uintp = uintp
c_int8 = int8
np_int8 = int8
c_int16 = int16
np_int16 = int16
c_int32 = int32
np_int32 = int32
c_int64 = int64
np_int64 = int64
c_intp = intp
py_int = intp
np_intp = intp
c_float16 = float16
np_float16 = float16
c_float32 = float32
np_float32 = float32
c_float64 = float64
py_float = float64
np_float64 = float64
np_complex64 = complex64
py_complex = complex128
np_complex128 = complex128
py_signed_domain = signed_domain
np_signed_domain = signed_domain
np_unsigned_domain = unsigned_domain
py_integer_domain = integer_domain
np_integer_domain = integer_domain
py_real_domain = real_domain
np_real_domain = real_domain
py_complex_domain = complex_domain
np_complex_domain = complex_domain
py_number_domain = number_domain
np_number_domain = number_domain
b1 = bool_
i1 = int8
i2 = int16
i4 = int32
i8 = int64
u1 = uint8
u2 = uint16
u4 = uint32
u8 = uint64
f2 = float16
f4 = float32
f8 = float64
c8 = complex64
c16 = complex128
np_float_ = float32
np_double = float64
double = float64
float_ = float32
char: Incomplete
uchar: Incomplete
short: Incomplete
ushort: Incomplete
int_: Incomplete
uint: Incomplete
long_: Incomplete
ulong: Incomplete
longlong: Incomplete
ulonglong: Incomplete
c_integer_domain = c_signed_domain | c_unsigned_domain
c_number_domain = c_real_domain | c_integer_domain | c_complex_domain
py_integer_domain = py_signed_domain
py_number_domain = py_real_domain | py_integer_domain | py_complex_domain
np_integer_domain = np_signed_domain | np_unsigned_domain
np_number_domain = np_real_domain | np_integer_domain | np_complex_domain
np_double = np_float64
