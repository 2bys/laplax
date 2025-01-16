"""Hessian vector product for curvature estimation."""

import jax
import jax.numpy as jnp

from laplax.enums import LossFn
from laplax.types import (
    Array,
    Callable,
    Data,
    Float,
    InputArray,
    Int,
    ModelFn,
    Num,
    Params,
    PyTree,
    TargetArray,
)
from laplax.util.tree import mul


def log_sigmoid_cross_entropy(
    logits: Num[Array, "..."], targets: Num[Array, "..."]
) -> Num[Array, "..."]:
    """Computes log sigmoid cross entropy given logits and targets.

    This function computes the cross entropy loss between the sigmoid of the logits
    and the target values. The formula implemented is:
    -targets * log_sigmoid(logits) - (1 - targets) * log_sigmoid(-logits)

    Args:
        logits: The predicted logits before sigmoid activation
        targets: The target values (0 or 1)

    Returns:
        The computed loss value
    """
    return -targets * jax.nn.log_sigmoid(logits) - (1 - targets) * jax.nn.log_sigmoid(
        -logits
    )


def hvp(func: Callable, primals: PyTree, tangents: PyTree) -> PyTree:
    r"""Compute the Hessian-vector product (HVP) for a given function.

    The Hessian-vector product is computed by differentiating the gradient of the
    function. This avoids explicitly constructing the Hessian matrix, making the
    computation efficient.

    Args:
        func: The scalar function for which the HVP is computed.
        primals: The point at which the gradient and Hessian are evaluated.
        tangents: The vector to multiply with the Hessian.

    Returns:
        The Hessian-vector product.
    """
    return jax.jvp(jax.grad(func), (primals,), (tangents,))[1]


def concatenate_model_and_loss_fn(
    model_fn: ModelFn,  # type: ignore[reportRedeclaration]
    loss_fn: LossFn | str | Callable,
    *,
    has_batch: bool = False,
) -> Callable[[InputArray, TargetArray, Params], Num[Array, "..."]]:
    r"""Combine a model function and a loss function into a single callable.

    This creates a new function that evaluates the model and applies the specified
    loss function. If `has_batch` is `True`, the model function is vectorized over
    the batch dimension using `jax.vmap`.

    Mathematically, the combined function computes:
    $L(x, y, \theta) = \text{loss}(f(x, \theta), y)$, where $f$ is the model function,
    $\theta$ are the model parameters, $x$ is the input, and $y$ is the target.

    Args:
        model_fn: The model function to evaluate.
        loss_fn: The loss function to apply. Supported options are:
            - `LossFn.MSE` for mean squared error.
            - `LossFn.CROSSENTROPY` for cross-entropy loss.
            - A custom callable loss function.
        has_batch: Whether the model function should be vectorized over the batch.

    Returns:
        A combined function that computes the loss for given inputs, targets, and
        parameters.
    """
    if has_batch:
        model_fn = jax.vmap(model_fn, in_axes=(0, None))

    if loss_fn == LossFn.MSE:

        def loss_wrapper(
            input: InputArray, target: TargetArray, params: Params
        ) -> Num[Array, "..."]:
            return jnp.sum((model_fn(input, params) - target) ** 2)

        return loss_wrapper

    if loss_fn == LossFn.CROSSENTROPY:

        def loss_wrapper(
            input: InputArray, target: TargetArray, params: Params
        ) -> Num[Array, "..."]:
            return log_sigmoid_cross_entropy(model_fn(input, params), target)

        return loss_wrapper

    if callable(loss_fn):

        def loss_wrapper(
            input: InputArray, target: TargetArray, params: Params
        ) -> Num[Array, "..."]:
            return loss_fn(model_fn(input, params), target)

        return loss_wrapper

    msg = f"unknown loss function: {loss_fn}"
    raise ValueError(msg)


def create_hessian_mv_without_data(
    model_fn: ModelFn,  # type: ignore[reportRedeclaration]
    params: Params,
    loss_fn: LossFn | str | Callable,
    factor: Float,
    *,
    has_batch: bool = False,
    **kwargs,
) -> Callable[[Params, Data], Params]:
    r"""Computes the Hessian-vector product (HVP) for a model and loss function.

    This function computes the HVP by combining the model and loss functions into a
    single callable. It evaluates the Hessian at the provided model parameters, with
    respect to the model and loss function.

    Mathematically: $H \cdot v = \nabla^2 L(x, y, \theta) \cdot v$, where $L$ is the
    combined loss function, $\theta$ are the parameters, and $v$ is the input vector.

    Args:
        model_fn: The model function to evaluate.
        params: The parameters of the model.
        loss_fn: The loss function to apply. Supported options are:
            - `LossFn.MSE` for mean squared error.
            - `LossFn.CROSSENTROPY` for cross-entropy loss.
            - A custom callable loss function.
        factor: Scaling factor for the Hessian computation.
        has_batch: Whether the model function should be vectorized over the batch.
        **kwargs: Additional arguments (ignored).

    Returns:
        A function that computes the HVP for a given vector and batch of data.
    """
    del kwargs

    new_model_fn: Callable[[InputArray, TargetArray, Params], Num[Array, "..."]]  # noqa: UP037

    new_model_fn = concatenate_model_and_loss_fn(model_fn, loss_fn, has_batch=has_batch)

    def _hessian_mv(vec: Params, data: Data) -> Params:
        return mul(
            factor,
            hvp(
                lambda p: new_model_fn(data["input"], data["target"], p),
                params,
                vec,
            ),
        )

    return _hessian_mv


def create_hessian_mv(
    model_fn: ModelFn,  # type: ignore[reportRedeclaration]
    params: Params,
    data: Data,
    loss_fn: LossFn | str | Callable,
    num_curv_samples: Int | None = None,
    num_total_samples: Int | None = None,
    **kwargs,
) -> Callable[[Params], Params]:
    r"""Computes the Hessian-vector product (HVP) for a model and loss fn. with data.

    This function wraps `create_hessian_mv_without_data`, fixing the dataset to produce
    a function that computes the HVP for the specified data.

    Mathematically: $H \cdot v = \nabla^2 L(x, y, \theta) \cdot v$, where $L$ is the
    combined loss function, $\theta$ are the parameters, and $v$ is the input vector of
    the HVP.

    Args:
        model_fn: The model function to evaluate.
        params: The parameters of the model.
        data: A batch of input and target data.
        loss_fn: The loss function to apply. Supported options are:
            - `LossFn.MSE` for mean squared error.
            - `LossFn.CROSSENTROPY` for cross-entropy loss.
            - A custom callable loss function.
        num_curv_samples: Number of samples used to calculate the Hessian. Defaults to
            None, in which case it is inferred from `data` as its batch size. Note that
            for losses that contain sums even for a single input (e.g., pixel-wise
            semantic segmentation losses), this number is _not_ the batch size.
        num_total_samples: Number of total samples the model was trained on. See the
            remark in `num_ggn_samples`'s description. Defaults to None, in which case
            it is set to equal `num_ggn_samples`.
        **kwargs: Additional arguments.

    Returns:
        A function that computes the HVP for a given vector and the fixed dataset.

    Note: The function assumes a batch dimension.
    """
    if num_curv_samples is None:
        num_curv_samples = data["input"].shape[0]

    if num_total_samples is None:
        num_total_samples = num_curv_samples

    curv_scaling_factor = num_total_samples / num_curv_samples

    hessian_mv = create_hessian_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=loss_fn,
        factor=curv_scaling_factor,
        **kwargs,
    )

    def wrapped_hessian_mv(vec: Params) -> Params:
        return hessian_mv(vec, data)

    return wrapped_hessian_mv
