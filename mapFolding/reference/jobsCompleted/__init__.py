"""
New Contribution to OEIS A001415: First-ever calculations for 2x19 and 2x20 maps

My first computation of the 2x19 map completed on 01/10/2025.
My first computation of the 2x20 map completed on 01/14/2025.

These represent the first-ever calculations of fold patterns for these dimensions
and extend the known values in the Online Encyclopedia of Integer Sequences (OEIS)
for series A001415 "Number of ways of folding a 2 X n strip of stamps".

Directory Structure:
--------------------------------------------------------------------------
(.venv) > dir mapFolding/reference/jobsCompleted

01/10/2025  02:27 AM                14 [2,19].foldsTotal  # First 2x19 calculation
01/14/2025  02:04 PM                15 [2,20].foldsTotal  # First 2x20 calculation
--------------------------------------------------------------------------

In the subfolder [2x19]:
--------------------------------------------------------------------------
(.venv) > dir mapFolding/reference/jobsCompleted/[2x19]

02/12/2025  03:48 PM            19,822 p2x19.py           # Optimized algorithm implementation
01/21/2025  03:36 AM            50,219 stateJob.pkl       # Serialized computation state
01/22/2025  07:15 AM                14 [2x19].foldsTotal  # Result of calculation
01/21/2025  03:37 AM         9,678,080 [2x19].ll          # LLVM IR code generated from the optimized algorithm
--------------------------------------------------------------------------

A version of the algorithm tuned to compute a 2x19 map took approximately 28 hours of computation.
The LLVM IR file ([2x19].ll) was generated using the getLLVMforNoReason module and provides
insight into the low-level optimizations that made this computation possible.

Alternative Implementation:
--------------------------------------------------------------------------
(.venv) > dir mapFolding/reference/jobsCompleted/p2x19

02/16/2025  07:01 PM                14 p2x19.foldsTotal   # Alternative implementation result
02/16/2025  12:40 AM             6,423 p2x19.py           # Alternative optimized algorithm
--------------------------------------------------------------------------

This alternative implementation took approximately 18 hours of computation and demonstrates
how code transformation and algorithm optimization can significantly reduce computation time.

To use these values in your own research, you can access them through the OEIS_for_n function:
```
from mapFolding import oeisIDfor_n
result = oeisIDfor_n('A001415', 19)  # For the 2x19 calculation
result = oeisIDfor_n('A001415', 20)  # For the 2x20 calculation
```
"""
