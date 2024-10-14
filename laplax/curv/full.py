"""Full GGN or Hessian curvature estimation."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree
from jax import jvp, grad, jacfwd, jacrev
from laplax.curv.cov import prec_to_scale
from laplax.curv.util import get_inflate_pytree_fn, flatten_pytree


def hvp(fn: callable, params, data):
    """Compute hessian vector product of model function.

    Args:
        fn: The function to estimate the Hessian of.
        funcion_input: The parameters at which to estimate the Hessian.
        v: The vector o calculate the hvp with
    Returns:
        The hessian vector prodct
    """

    fn_params = lambda x: fn(x, data)
    return jacfwd(jacrev(fn_params))


def to_dense(mvp, params):
    return mvp(params)


def flatten_hessian(hessian_pytree: PyTree, params_pytree: PyTree) -> jax.Array:
    """Flatten the Hessian matrix.

    Args:
        hessian_pytree: The Hessian matrix represented as a PyTree.
        params_pytree: The parameters represented as a PyTree.

    Returns:
        The flattened Hessian matrix.
    """
    # Tree flatten both hessian and params
    flatten_tree = jax.tree_util.tree_flatten(hessian_pytree)[0]
    flatten_params = jax.tree_util.tree_flatten(params_pytree)[0]

    # Concatenate hessian to tree
    n_parts = len(flatten_params)
    full_hessian = jnp.concatenate(
        [
            jnp.concatenate(
                [
                    arr.reshape(np.prod(p.shape), -1)
                    for arr in flatten_tree[i * n_parts: (i + 1) * n_parts]
                ],
                axis=1,
            )
            for i, p in enumerate(flatten_params)
        ],
        axis=0,
    )

    return full_hessian


def estimate_hessian(fn: callable, params, data):
    """Estimate the Hessian of a function at a given point.

    Args:
        fn: The function to estimate the Hessian of.
        params: The parameters at which to estimate the Hessian.
        data: The data to evaluate the function at.

    Returns:
        The estimated Hessian.
    """
    # Get the Hessian of the loss
    hessian_fn = jax.hessian(fn)

    # Estimate the Hessian
    hessian = hessian_fn(params, data)

    return hessian


def cov_scale_full_hessian(hessian: jax.Array, scaling, prior) -> jax.Array:  # noqa: D417
    """Scale the full Hessian by a given scale.

    Args:
        hessian: The full Hessian.
        scale: The scale to apply to the Hessian.

    Returns:
        The scaled full Hessian.
    """
    return prec_to_scale(scaling * hessian + prior)
