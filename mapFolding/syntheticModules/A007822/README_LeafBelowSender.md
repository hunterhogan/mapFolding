# LeafBelowSender: Passing Arrays from Numba to Python

## Overview

This module provides a mechanism to pass numpy arrays from within numba nopython mode functions to Python code for processing. It follows the same pattern as `numba_progress.ProgressBar`, enabling communication from jitted code to Python without forcing object mode fallback.

## Problem

In numba nopython mode, you cannot call regular Python functions. This is a problem when you need to pass data (like arrays) from a jitted function to Python code for processing. The naive approach of calling a Python function directly will either:
1. Fail with a typing error in nopython mode
2. Force the jit to use object mode, losing all performance benefits

## Solution

`LeafBelowSender` provides a numba-compatible way to send arrays:

1. **Shared Memory Buffer**: Uses a circular buffer (2D numpy array) accessible from both numba and Python
2. **Atomic Operations**: Thread-safe writes using atomic increment operations
3. **Background Processing**: Python thread reads from buffer and processes arrays asynchronously
4. **Type System Integration**: Full numba type system integration (unbox/box, overload_method)

## Usage Example

```python
from mapFolding.syntheticModules.A007822.leafBelowSender import LeafBelowSender
import numba
import numpy as np

# Create a sender with a processor function
results = []
sender = LeafBelowSender(
    buffer_size=1000,    # Number of array slots
    array_size=20,       # Size of each array
    processor_func=lambda arr: results.append(arr.copy())
)

# Use from numba-jitted code
@numba.jit(nopython=True)
def process_data(sender):
    for i in range(100):
        arr = np.array([i, i*2, i*3], dtype=np.int64)
        sender.push(arr)  # This works in nopython mode!

# Call the jitted function
process_data(sender)
sender.close()

print(f"Processed {len(results)} arrays")
```

## How It Works

### 1. Type System Integration

The sender is registered as a numba type:

```python
class LeafBelowSenderType(types.Type):
    """Numba type representation"""
    
@typeof_impl.register(LeafBelowSender)
def typeof_leafbelow_sender(val, c):
    return LeafBelowSenderTypeInstance
```

### 2. Unboxing

When passed to a jitted function, the Python object is "unboxed" to expose only the data numba can work with:

```python
@unbox(LeafBelowSenderType)
def unbox_leafbelow_sender(typ, obj, c):
    # Extract buffer, counters, etc. from Python object
    # Return native representation
```

### 3. Method Overloading

The `.push()` method is overloaded for numba:

```python
@overload_method(LeafBelowSenderType, "push", jit_options={"nogil": True})
def _overload_push(self, array):
    def push_impl(self, array):
        # Atomically get write position
        write_index = atomic_add_int64(self.items_pushed, 0, 1)
        
        # Copy to circular buffer
        slot = write_index % self.buffer_size
        # ... copy array elements ...
    
    return push_impl
```

### 4. Background Processing

A Python thread continuously reads from the buffer:

```python
def _process_arrays(self):
    while not self._exit_event.is_set():
        while self.items_processed[0] < self.items_pushed[0]:
            slot = self.items_processed[0] % self.buffer_size
            array_copy = self.buffer[slot, :].copy()
            self.processor_func(array_copy)
            self.items_processed[0] += 1
```

## Integration in asynchronousNumba.py

### Before

The original code tried to call a Python function directly:

```python
@jit(...)
def count(...):
    if leaf1ndex > leavesTotal:
        filterAsymmetricFolds(leafBelow)  # ERROR: Can't call Python function!
```

### After

Now it uses the sender:

```python
@jit(...)
def count(..., leafBelowSender):
    if leaf1ndex > leavesTotal:
        leafBelowSender.push(leafBelow)  # Works in nopython mode!

def doTheNeedful(state, maxWorkers):
    # Initialize sender
    initializeConcurrencyManager(maxWorkers)
    leafBelowSender = getLeafBelowSender()
    
    # Pass sender to jitted function
    count(..., leafBelowSender)
    
    # Get results
    result = getSymmetricFoldsTotal()
```

## Comparison with numba_progress

| Feature | numba_progress | LeafBelowSender |
|---------|----------------|-----------------|
| Purpose | Update progress counter | Pass arrays for processing |
| Shared State | Single int64 counter | 2D array buffer + counters |
| Method | `.update(n)` | `.push(array)` |
| Atomic Ops | Atomic add | Atomic add for index |
| Background Thread | Updates display | Processes arrays |

## Performance Considerations

- **Buffer Size**: Choose based on expected load. Too small = blocking, too large = memory waste
- **Array Size**: Must accommodate largest array. Smaller arrays are zero-padded
- **Processing Speed**: Background thread should keep up with push rate
- **Memory**: `buffer_size * array_size * 8 bytes` for int64 arrays

## Limitations

- Arrays must be convertible to int64 (or you can modify the implementation for other types)
- Fixed array size (padding/truncation as needed)
- Circular buffer can wrap if processor falls behind (older items overwritten)

## See Also

- [numba_progress](https://github.com/mortacious/numba-progress) - The inspiration for this implementation
- [Numba extending documentation](https://numba.pydata.org/numba-doc/latest/extending/index.html)
