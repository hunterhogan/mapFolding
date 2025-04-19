"""
Map Folding Test Suite and Validation Framework

This test suite provides comprehensive testing capabilities for the mapFolding package
and its optimization framework. It is specifically designed to enable both package
maintenance and custom extension testing, making it easy for users to validate their
own recipe configurations and job implementations.

## Key Testing Capabilities

1. **Algorithm Validation**
   - Tests core algorithm correctness against known OEIS sequence values
   - Validates both sequential and parallel execution paths
   - Ensures consistency across different implementation strategies

2. **Code Generation Testing**
   - Tests the AST transformation assembly line from source to optimized implementations
   - Validates that generated Numba-accelerated modules produce correct results
   - Ensures robust code generation across different parameter sets

3. **Job-Specific Testing**
   - Tests specialized job module generation for specific map shapes
   - Validates execution of the generated modules
   - Verifies correct output file creation and value storage

## Testing Your Own Implementations

This suite is designed to make it easy to test your custom recipes and jobs:

### For Custom Recipes (RecipeSynthesizeFlow):
Copy and adapt the `syntheticDispatcherFixture` and associated tests from
`test_computations.py` to validate your customized code transformation assembly lines.

### For Custom Jobs (RecipeJob):
Copy and adapt the `test_writeJobNumba` function to test specialized job modules
for specific map shapes with your custom configurations.

The entire test infrastructure is built on fixtures and utilities that handle
complex setup and validation, allowing you to focus on your implementation details
while leveraging the existing validation framework.

See the module docstrings in `test_computations.py` and `conftest.py` for detailed
guidance on adapting these tests for your own purposes.
"""
