"""This file contains a GGN class object, behaving like a linear operator.

This does not support pytree structures.
"""

import operator
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from laplax.config import laplax_dtype
from laplax.curv.util import flatten_pytree, get_inflate_pytree_fn


def flatten_model_jacobian(jacobian_tree: Any) -> jax.Array:
    """Flatten the model Jacobian."""
    jacobian_list = jax.tree.flatten(jacobian_tree)[0]

    return jnp.concatenate(
        [jacobian.reshape(jacobian.shape[0], -1) for jacobian in jacobian_list], axis=-1
    )  # [C, P]


def get_model_jacobian_fn(
    model_fn: Callable, params: dict
) -> Callable[[jax.Array], jax.Array]:
    jacobian_fn = jax.jacrev(model_fn)

    def flattened_jacobian_fn(x: jax.Array) -> jax.Array:
        return flatten_model_jacobian(jacobian_fn(params, x))

    return flattened_jacobian_fn


class GGN:
    """GGN class object."""

    def __init__(  # noqa: D107
        self,
        model_fn: Callable,
        params: list[dict] | list,
        data: tuple[jax.Array, jax.Array],
        dtype: jnp.dtype | None = None,
    ) -> None:
        self.params_flat, self.tree_def, self.shapes = flatten_pytree(params)
        num_params = self.params_flat.shape[0]
        self.params = params
        self.data = data
        self.model_fn = model_fn

        # self.astype(dtype)  # Maybe remove or make optional

        # Set helper functions
        self.inflate_pytree = jax.jit(get_inflate_pytree_fn(self.tree_def, self.shapes))
        self.jvp_fn = lambda params, x, vec: jax.jvp(
            lambda p: model_fn(params=p, input=x), (params,), (vec,)
        )
        self.vjp_fn = lambda params, x: jax.vjp(
            lambda p: model_fn(params=p, input=x), params
        )
        self.shape = (num_params, num_params)
        self.dtype = dtype or laplax_dtype()

    def mv_ggn_ptw(self, x: jax.Array, vec: jax.Array, params: dict) -> Any:
        inflated_vec = self.inflate_pytree(vec)
        mean_logit, Jv = self.jvp_fn(params, x, inflated_vec)  # [C], [C] # noqa: F841
        # H_loss = 2 #get_cross_entropy_loss_hessian(mean_logit)  # [C, C]
        H_loss_Jv = 2 * Jv  # [C]
        Gv = self.vjp_fn(params, x)[1](H_loss_Jv)[0]  # {[P]}

        return Gv

    def mv_ggn_global(self, x: jax.Array, vec: jax.Array, params: dict) -> Any:
        def body_fn(carry, x_i):
            mv_ggn_ptw_result = self.mv_ggn_ptw(x_i, vec, params)
            return jax.tree.map(operator.add, carry, mv_ggn_ptw_result), None

        num_samples = x.shape[0]
        initial_carry = jax.tree.map(jnp.zeros_like, params)
        sum_result = jax.lax.scan(body_fn, initial_carry, x)[0]
        mean_result = jax.tree.map(lambda arr: arr / num_samples, sum_result)

        return mean_result

    def mv_ggn_global_flatten(
        self, x: jax.Array, vec: jax.Array, params: dict
    ) -> jax.Array:
        out = flatten_pytree(self.mv_ggn_global(x, vec, params))[0]

        return out

    def mv(self, vec: jax.Array):
        return self.mv_ggn_global_flatten(self.data[0], vec, self.params)

    @partial(jax.jit, static_argnums=0)
    def __matmul__(self, vec: jax.Array) -> jax.Array:
        """Matrix-matrix product.

        This takes the locally implemented matrix-vector product (mv) and applies it to
        a matrix. Vectors are automatically shared. It contains also the logic for
        turning a matrix-vector product into a dense matrix.
        """
        if hasattr(self, "matmul"):  # This for the cases where mv supports matmul.
            return self.matmul(vec)

        if len(vec.shape) == 1:
            vec = vec[:, None]
            flatten = True
        else:
            flatten = False

        if vec.shape[-2] != self.shape[-1]:
            msg = f"expected vec.shape[-2] to be {self.shape[-1]}, got {vec.shape[-2]} instead."  # noqa: E501
            raise ValueError(msg)

        if len(vec.shape) > 2:
            msg = "Only 2D arrays are supported."
            raise ValueError(msg)

        if vec.dtype != self.dtype:
            msg = f"expected vec.dtype to be {self.dtype}, got {vec.dtype} instead."
            raise ValueError(msg)

        res = jax.lax.map(
            self.mv,
            vec.T,
            batch_size=1,
        ).T  # jax.lax.map shares over first axes.

        return res if not flatten else res[..., :, 0]

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.asarray(
            self.__matmul__(jnp.asarray(x, dtype=jnp.float32)), dtype=jnp.float64
        )
