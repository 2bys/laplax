import jax
import jax.numpy as jnp
import numpy as np
import pytest

from laplax.types import Callable, Iterable, KeyType
from laplax.util.loader import DataLoaderMV, reduce_add
from laplax.util.mv import diagonal, to_dense


def set_array_loader(
    shape: tuple[int, int] = (4, 4),
    batch_size: int = 8,
    num_batches: int = 5,
    key: KeyType | None = None,
) -> Callable[[], Iterable[jax.Array]]:
    """Create a data loader function that generates random batches.

    Args:
        shape: Shape of each matrix in the batch (rows, cols).
        batch_size: Number of matrices in each batch.
        num_batches: Number of batches to generate.
        key: Optional PRNG key for random number generation.

    Returns:
        A callable that returns an iterator yielding random batches of shape
        (batch_size, *shape).
    """

    def loader():
        key_local = key if key is not None else jax.random.key(0)
        for _ in range(num_batches):
            key_local, subkey = jax.random.split(key_local)
            # Generate random batch with the given shape
            x = jax.random.normal(subkey, (batch_size, *shape))
            yield x

    return loader


def single_batch_mv(vec: jax.Array, data: jax.Array) -> jax.Array:
    """Compute matrix-vector product over a single batch.

    Args:
        vec: Input vector of shape (n,).
        data: Batch of matrices of shape (batch_size, m, n).

    Returns:
        Result of shape (m,) representing the sum of batch @ vec over the batch
        dimension.
    """
    return (data @ vec).sum(axis=0)


def create_full_mv(
    loader: Callable[[], Iterable[jax.Array]], output_dim: int
) -> Callable[[jax.Array], jax.Array]:
    """Create a function that computes matrix-vector product over all batches.

    Args:
        loader: A callable that returns an iterator over batches.
        output_dim: Dimension of the output vector.

    Returns:
        A function that takes a vector and returns the matrix-vector product
        summed over all batches.
    """

    def mv(v):
        val = jnp.zeros(output_dim)
        for batch in loader():
            val += (batch @ v).sum(axis=0)
        return val

    return mv


def test_to_dense() -> None:
    """Test equivalence of to_dense between naive and DataLoaderMV implementations.

    Verifies that applying to_dense to a naive matrix-vector product function
    produces the same result as applying it to a DataLoaderMV instance.
    """
    shape = (4, 4)

    # Naive implementation using a loader
    loader_naive = set_array_loader(shape=shape)
    mv_naive = create_full_mv(loader_naive, shape[0])
    arr_naive = to_dense(mv_naive, layout=shape[0])

    # Implementation using DataLoaderMV for dispatching
    loader_dispatch = set_array_loader(shape=shape)
    data_mv = DataLoaderMV(
        single_batch_mv, loader_dispatch(), transform=lambda x: x, reduce=reduce_add
    )
    arr_dispatch = to_dense(data_mv, layout=shape[0])

    np.testing.assert_allclose(arr_naive, arr_dispatch, atol=1e-6, rtol=1e-6)


def test_diagonal() -> None:
    """Test equivalence of diagonal between naive and DataLoaderMV implementations.

    Verifies that:
    1. Applying diagonal to a naive matrix-vector product function produces
       the same result as applying it to a DataLoaderMV instance.
    2. The diagonal elements match the results of applying the matrix-vector
       product to one-hot vectors.
    """
    shape = (4, 4)

    # Naive diagonal computation
    loader_naive = set_array_loader(shape=shape)
    mv_naive = create_full_mv(loader_naive, shape[0])
    diag_naive = diagonal(mv_naive, layout=shape[0])

    # Diagonal computation using DataLoaderMV
    loader_dispatch = set_array_loader(shape=shape)
    data_mv = DataLoaderMV(
        single_batch_mv, loader_dispatch(), transform=lambda x: x, reduce=reduce_add
    )
    diag_dispatch = diagonal(data_mv, layout=shape[0])

    np.testing.assert_allclose(diag_naive, diag_dispatch, atol=1e-6, rtol=1e-6)

    # Check individual diagonal entries via one-hot inputs
    mv_instance = create_full_mv(set_array_loader(shape=shape), shape[0])
    for i in range(shape[0]):
        one_hot = jnp.zeros(shape[0]).at[i].set(1.0)
        expected = mv_instance(one_hot)[i]
        np.testing.assert_allclose(expected, diag_naive[i], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
