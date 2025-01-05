from typing import get_args

import pytest
import inspect
import random
import jax
import jax.numpy as jnp
from laplax.util.flatten import create_pytree_flattener
from pytest_cases import parametrize_with_cases
from laplax.util import tree  # Import your tree module
from .cases import case_random_pytree  # Import your cases
from .cases.case_random_pytree import case_random_pytree, case_two_pytree

import pytest
import inspect
from laplax.util.flatten import create_pytree_flattener
from pytest_cases import parametrize_with_cases
from laplax.util import tree  # Import your tree module


# Discover all functions in the tree module
tree_functions = inspect.getmembers(tree, inspect.isfunction)


# Parametrize single input functions with case_random_pytree
@pytest.mark.parametrize(
    "test_case",
    [case_random_pytree],  # Use case_random_pytree for single-input functions
)
@pytest.mark.parametrize(
    "func",
    [func for name, func in tree_functions if len(inspect.signature(func).parameters) == 1],  # Filter single-input functions
    ids=[name for name, func in tree_functions if len(inspect.signature(func).parameters) == 1]
)
def test_single_input_functions(test_case, func):
    """
    Test each single-input function in the tree module with a random PyTree.
    """
    pytree, vector = next(test_case())

    # Call the function with the single input
    result = func(pytree)
    assert result is not None, f"{func.__name__} returned None"


# Parametrize two-input functions with case_two_pytree
@pytest.mark.parametrize(
    "test_case",
    [case_two_pytree],  # Use case_two_pytree for two-input functions
)
@pytest.mark.parametrize(
    "func",
    [func for name, func in tree_functions if len(inspect.signature(func).parameters) == 2],  # Filter two-input functions
    ids=[name for name, func in tree_functions if len(inspect.signature(func).parameters) == 2]
)
def test_two_input_functions(test_case, func):
    """
    Test each two-input function in the tree module with random PyTrees.
    """
    pytree1, vector1 = next(test_case)
    pytree2, vector2 = next(test_case)

    # Call the function with two inputs
    result = func(pytree1, pytree2)
    flatten_result, _ = create_pytree_flattener(result)

    # You can customize the expected behavior for each function here
    # Example: Check for "add", "sub", or other operations you might be testing
    if func.__name__ == "add":
        expected_vector = vector1 + vector2
    elif func.__name__ == "sub":
        expected_vector = vector1 - vector2
    else:
        pytest.fail(f"Unknown behavior for function {func.__name__}")

    # Assert the flattened result matches the expected behavior
    assert jnp.allclose(flatten_result(result), expected_vector), (
        f"{func.__name__} failed for two PyTrees"
    )
