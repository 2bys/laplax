from typing import get_args

import pytest
import inspect
import random
import jax
import jax.numpy as jnp
from laplax.util.flatten import create_pytree_flattener
from pytest_cases import parametrize_with_cases
from laplax.util import tree  # Import your tree module
from .cases import case_generators  # Import your cases
from .cases.case_generators import case_random_pytree, case_two_pytree

import itertools
import pytest
import inspect
from laplax.util.flatten import create_pytree_flattener
from pytest_cases import parametrize_with_cases
from laplax.util import tree  # Import your tree module


# Discover all functions in the tree module
tree_functions = inspect.getmembers(tree, inspect.isfunction)
NUMBER_TEST_RUNS = 5

@pytest.mark.parametrize(
    "test_case",
    [next(case_random_pytree()) for _ in range(NUMBER_TEST_RUNS)],
)
@pytest.mark.parametrize(
    "func",
    [func for name, func in tree_functions if len(inspect.signature(func).parameters) == 1],
    ids=[name for name, func in tree_functions if len(inspect.signature(func).parameters) == 1]
)
def test_single_input_functions(test_case, func):
    """
    Test each single-input function in the tree module with a random PyTree.
    """
    pytree, vector, flatten = test_case
    int_tree = jax.tree_map(lambda x: x.astype(jnp.int32), pytree)
    name = func.__name__

    if name == "get_size":
        assert func(pytree) == len(vector)
    elif name == "ones_like":
        assert all(jax.tree_util.tree_map(lambda x: x == 1, func(pytree)))
    elif name == "zeros_like":
        assert all(jax.tree_util.tree_map(lambda x: x == 0, func(pytree)))
    elif name == "eye_like":
        assert jax.tree_util.tree_map(lambda x: jnp.allclose(x, jnp.eye(x.shape[0])), func(pytree))
    elif name == "invert":
        #TODO: change this in the test
        result = func(int_tree)
        assert flatten(result) == jnp.invert(vector.astype(jnp.int32))
    elif name == "neg":
        assert flatten(func(pytree)) == -vector
    elif name == "sqrt":
        flatten(func(pytree)) == jnp.sqrt(vector)
    else:
        pytest.fail(f"Unknown behavior for function {name}")



@pytest.mark.parametrize(
    "test_case",
    [next(case_two_pytree()) for _ in range(NUMBER_TEST_RUNS)],
)
@pytest.mark.parametrize(
    "func",
    [func for name, func in tree_functions if len(inspect.signature(func).parameters) == 2],
    ids=[name for name, func in tree_functions if len(inspect.signature(func).parameters) == 2]
)
def test_two_input_functions(test_case, func):
    """
    Test each two-input function in the tree module with random PyTrees.
    """
    pytree1, vector1 = test_case[0]
    pytree2, vector2 = test_case[1]

    result = func(pytree1, pytree2)
    flatten_result, _ = create_pytree_flattener(result)

    if func.__name__ == "add":
        expected_vector = vector1 + vector2
    elif func.__name__ == "sub":
        expected_vector = vector1 - vector2
    else:
        pytest.fail(f"Unknown behavior for function {func.__name__}")

    assert jnp.allclose(flatten_result(result), expected_vector), (
        f"{func.__name__} failed for two PyTrees"
    )
