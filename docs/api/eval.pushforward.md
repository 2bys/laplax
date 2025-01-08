# Pushforward predictives

## Linearized Laplace

The linearized Laplace approximation propagates uncertainty through a first-order Taylor expansion of the model around the MAP estimate. The following functions implement this approach:

### Setup Functions

::: laplax.eval.pushforward.set_lin_pushforward


### Core Functions

::: laplax.eval.pushforward.lin_setup

::: laplax.eval.pushforward.lin_pred

::: laplax.eval.pushforward.lin_pred_var

::: laplax.eval.pushforward.lin_pred_std

::: laplax.eval.pushforward.lin_pred_cov

::: laplax.eval.pushforward.lin_samples


### Posterior GP Kernel

::: laplax.eval.pushforward.set_posterior_gp_kernel


## Non-linear Laplace

The non-linear Laplace approximation propagates uncertainty by sampling from the weight-space posterior and evaluating the model with these weight samples. The following functions implement this approach:

### Setup Functions

::: laplax.eval.pushforward.set_nonlin_pushforward


### Core Functions

::: laplax.eval.pushforward.nonlin_setup

::: laplax.eval.pushforward.nonlin_pred_mean

::: laplax.eval.pushforward.nonlin_pred_var

::: laplax.eval.pushforward.nonlin_pred_std

::: laplax.eval.pushforward.nonlin_pred_cov

::: laplax.eval.pushforward.nonlin_samples
