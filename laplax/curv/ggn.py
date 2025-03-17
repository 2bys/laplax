"""Generalized Gauss-Newton matrix-vector product and loss hessian."""

from collections.abc import Callable

import jax

from laplax.curv.hessian import hvp
from laplax.enums import LossFn
from laplax.types import (
    Array,
    Data,
    Float,
    Int,
    ModelFn,
    Num,
    Params,
    PredArray,
    TargetArray,
)
from laplax.util.tree import mul

# ---------------------------------------------------------------------
# Loss Hessian
# ---------------------------------------------------------------------


def _cross_entropy_hessian_mv(
    jv: PredArray, pred: PredArray, **kwargs
) -> Num[Array, "..."]:
    r"""Compute the Hessian-vector product for cross-entropy loss.

    This calculation uses the softmax probabilities of the predictions to compute the
    diagonal and off-diagonal components of the Hessian. The result is the difference
    between the diagonal contribution and the off-diagonal contribution of the Hessian.

    Mathematically, the Hessian-vector product is computed as:
    $H \cdot jv = \text{diag}(p) \cdot jv - p \cdot (p^\top \cdot jv), $ s
    where $p = \text{softmax}(\text{pred})$.

    Args:
        jv: Vector to multiply with the Hessian.
        pred: Model predictions (logits).
        **kwargs: Additional arguments (ignored).

    Returns:
        Hessian-vector product for cross-entropy loss.
    """
    del kwargs
    prob = jax.nn.softmax(pred)
    off_diag_jv = prob * (prob.reshape(1, -1) @ jv)
    diag_jv = prob * jv
    return diag_jv - off_diag_jv


def _mse_hessian_mv(jv: PredArray, **kwargs) -> PredArray:
    r"""Compute the Hessian-vector product for mean squared error loss.

    The Hessian of the mean squared error loss is a constant diagonal matrix with
    2 along the diagonal. Thus, the Hessian-vector product is simply 2 times the
    input vector.

    Mathematically:
    $H \cdot jv = 2 \cdot jv$.

    Args:
        jv: Vector to multiply with the Hessian.
        **kwargs: Additional arguments (ignored).

    Returns:
        Hessian-vector product for MSE loss.
    """
    del kwargs
    return 2 * jv


def create_loss_hessian_mv(
    loss_fn: LossFn | str | Callable[[PredArray, TargetArray], Num[Array, "..."]],
) -> Callable:
    r"""Create a function to compute the Hessian-vector product for a specified loss fn.

    For predefined loss functions like cross-entropy and mean squared error, the
    function computes their corresponding Hessian-vector products using efficient
    formulations. For custom loss functions, the Hessian-vector product is computed via
    automatic differentiation.

    Args:
        loss_fn: Loss function to compute the Hessian-vector product for. Supported
        options are:
            - "cross_entropy" for cross-entropy loss.
            - "mse" for mean squared error loss.
            - A custom callable loss function that takes predictions and targets.

    Returns:
        A function that computes the Hessian-vector product for the given loss function.
    """
    if loss_fn == LossFn.CROSSENTROPY:
        return _cross_entropy_hessian_mv

    if loss_fn == LossFn.MSE:
        return _mse_hessian_mv

    if isinstance(loss_fn, Callable):

        def custom_hessian_mv(
            jv: PredArray, pred: PredArray, target: TargetArray, **kwargs
        ) -> Num[Array, "..."]:
            del kwargs

            def loss_fn_local(p):
                return loss_fn(p, target)

            return hvp(loss_fn_local, pred, jv)

        return custom_hessian_mv

    msg = "unsupported loss function provided"
    raise ValueError(msg)


# -----------------------------------------------------------------------------------
# GGN Matrix-vector product factories
# -----------------------------------------------------------------------------------


def create_ggn_mv_without_data(
    model_fn: ModelFn,
    params: Params,
    loss_fn: LossFn | str | Callable,
    factor=Float,
    *,
    has_batch: bool = True,
) -> Callable[[Params, Data], Params]:
    r"""Create Generalized Gauss-Newton (GGN) matrix-vector productwithout fixed data.

    The GGN matrix is computed using the Jacobian of the model and the Hessian of the
    loss function. The resulting product is given by:
    $\text{factor} \cdot \sum_i J_i^\top H_{L, i} J_i \cdot v$
    where $J_i$ is the Jacobian of the model at data point $i$, $H_{L, i}$ is the
    Hessian of the loss, and $v$ is the vector.

    This function computes the above expression efficiently without hardcoding the
    dataset, making it suitable for distributed or batched computations.

    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        loss_fn: Loss function to use for the GGN computation.
        factor: Scaling factor for the GGN computation.
        has_batch: Whether the data has a batch dimension.

    Returns:
        A function that takes a vector and a batch of data, and computes the GGN
        matrix-vector product.

    Note:
        The function assumes that the data has a batch dimension.
    """
    # Create loss Hessian-vector product
    loss_fn_hessian_mv = create_loss_hessian_mv(loss_fn)
    if has_batch:
        loss_fn_hessian_mv = jax.vmap(loss_fn_hessian_mv)

    def ggn_mv(vec, batch):
        # Step 1: Single jvp for entire batch, if has_batch is True
        def fwd(p):
            if has_batch:
                return jax.vmap(lambda x: model_fn(input=x, params=p))(batch["input"])
            return model_fn(input=batch["input"], params=p)

        # Step 2: Linearize the forward pass
        z, jvp = jax.linearize(fwd, params)

        # Step 3: Compute J^T H J v
        HJv = loss_fn_hessian_mv(jvp(vec), pred=z, target=batch["target"])

        # Step 4: Compute the GGN vector
        arr = jax.linear_transpose(jvp, vec)(HJv)[0]

        return mul(factor, arr)

    return ggn_mv


def create_ggn_mv(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    loss_fn: LossFn | str | Callable,
    num_curv_samples: Int | None = None,
    num_total_samples: Int | None = None,
) -> Callable[[Params], Params]:
    r"""Computes the Generalized Gauss-Newton (GGN) matrix-vector product with data.

    The GGN matrix is computed using the Jacobian of the model and the Hessian of the
    loss function. For a given dataset, the GGN matrix-vector product is computed as:
    $\text{factor} \sum_{i=1}^N J_i^\top H_{L, i} J_i \cdot v$
    where $J_i$ is the Jacobian of the model for the $i$-th data point, $H_{L, i}$ is
    the Hessian of the loss for the $i$-th data point, and $N$ is the number of data
    points.

    This function hardcodes the dataset, making it ideal for scenarios where the dataset
    remains fixed.

    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        data: A batch of input and target data.
        loss_fn: Loss function to use for the GGN computation.
        # loss_scaling_factor: Factor by which the user-provided loss function is
        #     scaled. Defaults to 1.0.
        num_curv_samples: Number of samples used to calculate the GGN. Defaults to None,
            in which case it is inferred from `data` as its batch size. Note that for
            losses that contain sums even for a single input (e.g., pixel-wise semantic
            segmentation losses), this number is _not_ the batch size.
        num_total_samples: Number of total samples the model was trained on. See the
            remark in `num_ggn_samples`'s description. Defaults to None, in which case
            it is set to equal `num_ggn_samples`.

    Returns:
        A function that takes a vector and computes the GGN matrix-vector product for
        the given data.

    Note: The function assumes a batch dimension.
    """
    if num_curv_samples is None:
        num_curv_samples = data["input"].shape[0]

    if num_total_samples is None:
        num_total_samples = num_curv_samples

    curv_scaling_factor = num_total_samples / num_curv_samples

    ggn_mv = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=loss_fn,
        factor=curv_scaling_factor,
    )

    def wrapped_ggn_mv(vec: Params) -> Params:
        return ggn_mv(vec, data)

    return wrapped_ggn_mv
