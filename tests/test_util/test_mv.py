import jax
import jax.numpy as jnp
import numpy as np
import pytest

from laplax.util.flatten import create_pytree_flattener, wrap_function
from laplax.util.mv import diagonal, to_dense
from laplax.util.tree import get_size


@pytest.mark.parametrize("n", [2, 5])
def test_diagonal_dense(n):
    key = jax.random.PRNGKey(123)
    A = jax.random.normal(key, (n, n))

    # Calling diagonal on a dense matrix should return jnp.diag(A)
    diag_computed = diagonal(A)
    diag_expected = jnp.diag(A)
    np.testing.assert_allclose(diag_computed, diag_expected, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("n", [2, 5])
def test_diagonal_and_to_dense_flat_mvp(n):
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (n, n))

    def mv(x):
        return A @ x

    # 1) Compare diagonal(mv, layout=n) to jnp.diag(A)
    diag_computed = diagonal(mv, layout=n)
    diag_expected = jnp.diag(A)
    np.testing.assert_allclose(diag_computed, diag_expected, atol=1e-7, rtol=1e-7)

    # 2) Compare to_dense(mv, layout=n) to A
    dense_computed = to_dense(mv, layout=n)
    np.testing.assert_allclose(dense_computed, A, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize(("n1", "n2"), [(2, 3), (3, 4)])
def test_diagonal_and_to_dense_pytree_mvp(n1, n2):
    key = jax.random.PRNGKey(999)
    layout = {
        "x": jnp.zeros(n1),
        "y": jnp.zeros(n2),
    }
    example_flat, tree_def = jax.tree.flatten(layout)
    sizes = [leaf.size for leaf in example_flat]
    total_dim = sum(sizes)

    # Create a random (total_dim x total_dim) matrix A
    A = jax.random.normal(key, (total_dim, total_dim))

    def mv(pytree_vec):
        leaves, _ = jax.tree.flatten(pytree_vec)
        x_flat = jnp.concatenate([leaf.ravel() for leaf in leaves])
        y_flat = A @ x_flat

        # Split back into the shapes
        split_indices = np.cumsum(sizes[:-1])
        y_splits = jnp.split(y_flat, split_indices)

        # Reshape each split to match original leaf shape
        y_leaves = []
        for s, leaf_shape in zip(
            y_splits, [leaf.shape for leaf in example_flat], strict=True
        ):
            y_leaves.append(s.reshape(leaf_shape))

        # Unflatten back
        return jax.tree.unflatten(tree_def, y_leaves)

    # 1) Compare diagonal(mv, layout=layout) to jnp.diag(A)
    diag_computed = diagonal(mv, layout)
    diag_expected = jnp.diag(A)
    np.testing.assert_allclose(diag_computed, diag_expected, atol=1e-7, rtol=1e-7)

    # 2) Compare to_dense(mv, layout=layout) to A
    dense_computed = to_dense(mv, layout)

    flatten, unflatten = create_pytree_flattener(layout)
    np.testing.assert_allclose(flatten(dense_computed).reshape(*A.shape), A)

    mv_wrapped = wrap_function(mv, input_fn=unflatten, output_fn=flatten)
    dense_computed = to_dense(mv_wrapped, get_size(layout))
    flatten, unflatten = create_pytree_flattener(layout)
    np.testing.assert_allclose(flatten(dense_computed).reshape(*A.shape), A)
