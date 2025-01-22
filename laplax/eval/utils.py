"""Pushforward utilities for evaluating probabilistic predictions on datasets.

This module provides utilities for evaluating probabilistic models on datasets and
managing metric computations.

Key features include:
- Wrapping functions to store outputs in a structured format.
- Finalizing multiple functions and collecting results in a dictionary.
- Applying prediction functions across datasets to generate predictions and evaluating
  them against their targets.
- Computing and transforming evaluation metrics for datasets using custom or default
  metrics.

These utilities streamline dataset evaluation workflows and ensure flexibility in metric
computation and result aggregation.
"""

from laplax.types import Any, Array, Callable, Data, InputArray
from laplax.util.ops import lmap
from laplax.util.utils import identity

# Currently deprecated.
# def finalize_function_wrapper(
#     fn: Callable,
# ) -> Callable:
#     """Wrap a function to store its result in a dictionary.

#     This wrapper allows a function to be executed with specified arguments, and
#     its output is stored in the `results` dictionary under a specified name.

#     Args:
#         fn: A callable function to be wrapped.

#     Returns:
#         Callable: A wrapped function that takes `results`, `aux`, `name`, and
#         other keyword arguments, and updates the `results` dictionary.
#     """

#     def wrapper(
#         results: dict[str, Array], aux: dict[str, Any] | None, name: str, **kwargs
#     ):
#         results[name] = fn(**kwargs)
#         return results, aux

#     return wrapper


def finalize_functions(
    functions: list[Callable],
    results: dict,  # Typing must allow empty dict for initializations
    aux: dict[str, Any] | None = None,
    **kwargs,
) -> dict:
    """Execute a set of functions and store their results in a dictionary.

    This function iterates over a dictionary of functions, executes each
    function with the provided keyword arguments, and updates the `results`
    dictionary with their outputs.

    Args:
        functions: A dictionary where keys are names for the results, and values
            are callables to execute.
        results: A dictionary to store the outputs of the functions.
        aux: Auxiliary data passed to the functions.
        **kwargs: Additional arguments passed to each function.

    Returns:
        The updated `results` dictionary containing the outputs of all
        executed functions.
    """
    for func in functions:
        results, aux = func(results=results, aux=aux, **kwargs)
    return results


def evaluate_on_dataset(
    pred_fn: Callable[[InputArray], dict[str, Array]], data: Data, **kwargs
) -> dict:
    """Evaluate a prediction function on a dataset.

    This function applies a probabilistic predictive function (`pred_fn`) to
    each data point in the dataset, combining the predictions with the target
    labels.

    Args:
        pred_fn: A callable that takes an input array and returns predictions
            as a dictionary.
        data: A dataset, where each data point is a dictionary containing
            "input" and "target".
        **kwargs: Additional arguments, including:
            - `lmap_eval`: Batch size for processing data (default: "data").

    Returns:
        A dictionary containing predictions and target labels for the entire dataset.
    """

    def evaluate_data_point(dp: Data) -> dict[str, Array]:
        return {**pred_fn(dp["input"]), "target": dp["target"]}

    return lmap(evaluate_data_point, data, batch_size=kwargs.get("lmap_eval", "data"))


def apply_function(func, name="nll", field="results", **kwargs):
    def apply(results, aux, **local_kwargs):
        # Create key-value pair for function
        key_value_pairs = {}

        for k, v in kwargs.items():
            if v in results:
                key_value_pairs[k] = results[v]
            elif v in aux:
                key_value_pairs[k] = aux[v]
            else:
                msg = f"Key {k} not found in results or aux."
                raise ValueError(msg)

        res = func(**key_value_pairs, **local_kwargs)

        if field == "results":
            results[name] = res
        elif field == "aux":
            aux[name] = res
        else:
            msg = f"Field {field} must be either 'results' or 'aux'."
            raise ValueError(msg)

        return results, aux

    return apply


def transfer_entry(mapping: dict[str, str], field="results", access_from="aux"):
    def transfer(results, aux, **kwargs):
        del kwargs
        options = {"results": results, "aux": aux}
        if field == "results":
            for k, v in mapping.items():
                results[k] = options[access_from][v]
        elif field == "aux":
            for k, v in mapping.items():
                aux[k] = options[access_from][v]
        else:
            msg = f"Field {field} must be either 'results' or 'aux'."
            raise ValueError(msg)

        return results, aux

    return transfer


def evaluate_metrics_on_dataset(
    pred_fn: Callable[[InputArray], dict[str, Array]],
    data: Data,
    *,
    metrics: list[Callable],
    apply: Callable = identity,
    **kwargs,
) -> dict:
    """Evaluate a set of metrics on a dataset.

    This function computes specified metrics for predictions generated by a
    probabilistic predictive function (`pred_fn`) over a dataset. The results
    can optionally be transformed using an `apply` function.

    Args:
        pred_fn: A callable that takes an input array and returns predictions
            as a dictionary.
        data: A dataset, where each data point is a dictionary containing
            "input" and "target".
        metrics: A dictionary of metrics to compute, where keys are metric
            names and values are callables.
        apply: A callable to transform the evaluated metrics (default: identity).
        **kwargs: Additional arguments, including:
            - `lmap_eval_metrics`: Batch size for processing data (default: "data").

    Returns:
        dict: A dictionary containing the evaluated metrics for the entire
        dataset.
    """
    # Wrap metrics
    # metrics = {name: finalize_function_wrapper(fn) for name, fn in metrics.items()}

    # Setup pointwise evaluation
    def evaluate_data_point(dp: Data) -> dict[str, Array]:
        pred = {**pred_fn(dp["input"]), "target": dp["target"]}
        return finalize_functions(functions=metrics, results={}, aux=pred, **kwargs)

    # Evaluate metrics
    evaluated_metrics = lmap(
        evaluate_data_point, data, batch_size=kwargs.get("lmap_eval_metrics", "data")
    )
    return {metric: apply(evaluated_metrics[metric]) for metric in evaluated_metrics}
