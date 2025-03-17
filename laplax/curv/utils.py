"""Utility functions for curvature estimation."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from laplax.types import Array, Float, Num
from laplax.util.flatten import create_pytree_flattener, wrap_function
from laplax.util.tree import get_size

# -----------------------------------------------------------------------------
# Low-rank terms
# -----------------------------------------------------------------------------


@dataclass
class LowRankTerms:
    """Components of the low-rank curvature approximation.

    This dataclass encapsulates the results of the low-rank approximation, including
    the eigenvectors, eigenvalues, and a scalar factor which can be used for the prior.

    Attributes:
        U: Matrix of eigenvectors, where each column corresponds to an eigenvector.
        S: Array of eigenvalues associated with the eigenvectors.
        scalar: Scalar factor added to the matrix during the approximation.
    """

    U: Num[Array, "P R"]
    S: Num[Array, " R"]
    scalar: Float[Array, ""]


jax.tree_util.register_pytree_node(
    LowRankTerms,
    lambda node: ((node.U, node.S, node.scalar), None),
    lambda _, children: LowRankTerms(U=children[0], S=children[1], scalar=children[2]),
)


def get_matvec(A, *, layout=None, jit=True):
    """Returns a function that computes the matrix-vector product.

    Args:
        A: Either a jnp.ndarray or a callable performing the operation.
        layout: Required if A is callable; ignored if A is an array.
        jit: Whether to jit-compile the operator.

    Returns:
        A tuple (matvec, input_dim) where matvec is the callable operator.
    """
    if isinstance(A, jnp.ndarray):
        size = A.shape[0]

        def matvec(x):
            return A @ x

    else:
        matvec = A

        if layout is None:
            msg = "For a callable A, please provide `layout` in PyTree or int format."
            raise TypeError(msg)
        if isinstance(layout, int):
            size = layout
        else:
            try:
                flatten, unflatten = create_pytree_flattener(layout)
                matvec = wrap_function(matvec, input_fn=unflatten, output_fn=flatten)
                size = get_size(layout)
            except (ValueError, TypeError) as exc:
                msg = (
                    "For a callable A, please provide `layout` in PyTree or int format."
                )
                raise TypeError(msg) from exc

    if jit:
        matvec = jax.jit(matvec)
    return matvec, size
