"""LeafBelowSender: Pass arrays from numba-jitted code to Python for processing.

This module provides a mechanism to pass numpy arrays from within numba nopython mode
to Python code for processing, using a pattern similar to numba_progress.

The implementation uses:
- A circular buffer in shared memory (2D numpy array)
- Atomic counters for thread-safe read/write coordination
- A background thread that processes arrays as they arrive
- Numba type system integration for nopython mode compatibility

Inspired by: https://github.com/mortacious/numba-progress
"""

from mapFolding import Array1DLeavesTotal
from threading import Event, Lock, Thread
import numpy

from numba.core import cgutils, types
from numba.core.boxing import unbox_array
from numba.extending import (
	as_numba_type, box, intrinsic, make_attribute_wrapper, models, overload_method, 
	register_model, typeof_impl, unbox, NativeValue
)


# Atomic add intrinsic for int64 arrays
@intrinsic
def atomic_add_int64(typingctx, array_type, index_type, value_type):
	"""Atomically add value to array[index] and return the old value."""
	sig = types.int64(array_type, index_type, value_type)
	
	def atomic_add_impl(context, builder, sig, args):
		[array, index, value] = args
		array_val = context.make_array(sig.args[0])(context, builder, array)
		ptr = builder.gep(array_val.data, [index])
		old = builder.atomic_rmw('add', ptr, value, 'monotonic')
		return old
	
	return sig, atomic_add_impl


class LeafBelowSender:
	"""Send numpy arrays from numba-jitted code to Python for processing.
	
	This class provides a thread-safe way to pass arrays from numba nopython mode
	functions to Python code. It uses a circular buffer with atomic operations.
	
	Parameters
	----------
	buffer_size : int
		Number of array slots in the circular buffer (default: 1000)
	array_size : int
		Size of each array (number of elements)
	processor_func : callable
		Function to call with each array: processor_func(array) -> None
	"""
	
	def __init__(self, buffer_size: int, array_size: int, processor_func):
		self.buffer_size = buffer_size
		self.array_size = array_size
		self.processor_func = processor_func
		
		# Circular buffer: buffer[slot_index, element_index]
		self.buffer = numpy.zeros((buffer_size, array_size), dtype=numpy.int64)
		
		# Counter for items pushed (atomically incremented)
		self.items_pushed = numpy.zeros(1, dtype=numpy.int64)
		
		# Counter for items processed (modified by processor thread)
		self.items_processed = numpy.zeros(1, dtype=numpy.int64)
		
		# Lock for the processor (Python-side only, not used in numba)
		self._lock = Lock()
		
		# Event to signal shutdown
		self._exit_event = Event()
		
		# Start background processor thread
		self._processor_thread = Thread(target=self._process_arrays, daemon=True)
		self._processor_thread.start()
	
	def push(self, array: Array1DLeavesTotal) -> None:
		"""Push an array to be processed.
		
		This method is designed to be called from numba-jitted code.
		It copies the array into the circular buffer.
		
		Parameters
		----------
		array : Array1DLeavesTotal
			The array to push for processing
		"""
		# This will be overridden by numba's overload_method
		# For Python-side calls (testing), implement a simple version
		with self._lock:
			slot = self.items_pushed[0] % self.buffer_size
			size = min(array.size, self.array_size)
			self.buffer[slot, :size] = array[:size]
			self.items_pushed[0] += 1
	
	def close(self):
		"""Stop the background processor and wait for completion."""
		# Signal shutdown
		self._exit_event.set()
		
		# Wait for thread to finish
		self._processor_thread.join()
		
		# Process any remaining items
		self._drain_remaining()
	
	def _drain_remaining(self):
		"""Process all remaining items in the buffer."""
		while self.items_processed[0] < self.items_pushed[0]:
			slot = self.items_processed[0] % self.buffer_size
			array_copy = self.buffer[slot, :self.array_size].copy()
			self.processor_func(array_copy)
			self.items_processed[0] += 1
	
	def _process_arrays(self):
		"""Background thread function that processes arrays from the buffer."""
		while not self._exit_event.is_set():
			# Process all available items
			while self.items_processed[0] < self.items_pushed[0]:
				# Check if we should exit
				if self._exit_event.is_set():
					break
				
				# Get the slot to read from
				slot = self.items_processed[0] % self.buffer_size
				
				# Copy the array and process it
				array_copy = self.buffer[slot, :self.array_size].copy()
				self.processor_func(array_copy)
				
				# Mark as processed
				self.items_processed[0] += 1
			
			# Brief sleep to avoid busy-waiting
			self._exit_event.wait(0.001)
	
	def __enter__(self):
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()


# ============================================================================
# Numba Type System Integration
# ============================================================================

class LeafBelowSenderType(types.Type):
	"""Numba type representation for LeafBelowSender."""
	
	def __init__(self):
		super().__init__(name='LeafBelowSender')


# Create the type instance
LeafBelowSenderTypeInstance = LeafBelowSenderType()


@typeof_impl.register(LeafBelowSender)
def typeof_leafbelow_sender(val, c):
	"""Tell numba what type a LeafBelowSender instance is."""
	return LeafBelowSenderTypeInstance


as_numba_type.register(LeafBelowSender, LeafBelowSenderTypeInstance)


@register_model(LeafBelowSenderType)
class LeafBelowSenderModel(models.StructModel):
	"""Numba model defining what data is accessible from numba code."""
	
	def __init__(self, dmm, fe_type):
		members = [
			('buffer', types.Array(types.int64, 2, 'C')),
			('items_pushed', types.Array(types.int64, 1, 'C')),
			('items_processed', types.Array(types.int64, 1, 'C')),
			('buffer_size', types.int64),
			('array_size', types.int64),
		]
		models.StructModel.__init__(self, dmm, fe_type, members)


# Make attributes accessible in numba code
make_attribute_wrapper(LeafBelowSenderType, 'buffer', 'buffer')
make_attribute_wrapper(LeafBelowSenderType, 'items_pushed', 'items_pushed')
make_attribute_wrapper(LeafBelowSenderType, 'items_processed', 'items_processed')
make_attribute_wrapper(LeafBelowSenderType, 'buffer_size', 'buffer_size')
make_attribute_wrapper(LeafBelowSenderType, 'array_size', 'array_size')


@unbox(LeafBelowSenderType)
def unbox_leafbelow_sender(typ, obj, c):
	"""Convert Python LeafBelowSender object to numba native representation."""
	# Extract attributes from the Python object
	buffer_obj = c.pyapi.object_getattr_string(obj, 'buffer')
	items_pushed_obj = c.pyapi.object_getattr_string(obj, 'items_pushed')
	items_processed_obj = c.pyapi.object_getattr_string(obj, 'items_processed')
	buffer_size_obj = c.pyapi.object_getattr_string(obj, 'buffer_size')
	array_size_obj = c.pyapi.object_getattr_string(obj, 'array_size')
	
	# Create the native struct
	sender = cgutils.create_struct_proxy(typ)(c.context, c.builder)
	
	# Unbox arrays
	sender.buffer = unbox_array(types.Array(types.int64, 2, 'C'), buffer_obj, c).value
	sender.items_pushed = unbox_array(types.Array(types.int64, 1, 'C'), items_pushed_obj, c).value
	sender.items_processed = unbox_array(types.Array(types.int64, 1, 'C'), items_processed_obj, c).value
	
	# Unbox scalars
	sender.buffer_size = c.pyapi.long_as_longlong(buffer_size_obj)
	sender.array_size = c.pyapi.long_as_longlong(array_size_obj)
	
	# Clean up Python references
	c.pyapi.decref(buffer_obj)
	c.pyapi.decref(items_pushed_obj)
	c.pyapi.decref(items_processed_obj)
	c.pyapi.decref(buffer_size_obj)
	c.pyapi.decref(array_size_obj)
	
	# Check for errors
	is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
	return NativeValue(sender._getvalue(), is_error=is_error)


@box(LeafBelowSenderType)
def box_leafbelow_sender(typ, val, c):
	"""Cannot convert native representation back to Python object."""
	raise TypeError(
		"Native representation of LeafBelowSender cannot be converted back to a "
		"Python object as it contains internal Python state."
	)


@overload_method(LeafBelowSenderType, "push", jit_options={"nogil": True})
def _overload_push(self, array):
	"""Numba implementation of the push method."""
	if isinstance(self, LeafBelowSenderType):
		def push_impl(self, array):
			# Atomically get the next write slot
			write_index = atomic_add_int64(self.items_pushed, types.int64(0), types.int64(1))
			
			# Calculate slot in circular buffer
			slot = write_index % self.buffer_size
			
			# Copy array elements into buffer
			# Use min to handle arrays smaller than buffer size
			size = array.size if array.size < self.array_size else self.array_size
			i = 0
			while i < size:
				self.buffer[slot, i] = array[i]
				i += 1
			
			# Zero out remaining elements if array is smaller
			while i < self.array_size:
				self.buffer[slot, i] = 0
				i += 1
		
		return push_impl
