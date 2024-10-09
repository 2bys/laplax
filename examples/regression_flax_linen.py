import sys
import os
from jax.experimental.sparse import BCOO
import jax
from scipy.sparse import csr_matrix
import pickle

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from helper import get_sinusoid_example
import matplotlib.pyplot as plt

# jax
from jax import random, tree_util, value_and_grad, numpy as jnp
from jaxtyping import PyTree

# laplacestuff
from functools import partial
from laplax.curv.full import estimate_hessian, todense, flatten_hessian, cov_scale_full_hessian, hvp
from laplax.eval.push_forward import create_mc_predictions_for_data_point_fn

# flax for training, optax for optimization
from flax import linen as nn
import optax


# generate training data
n_data = 150
sigma_noise = 0.3
batch_size = 150
rng_key = random.key(711)

X_train, y_train, train_loader, X_test = get_sinusoid_example(n_data, sigma_noise, batch_size, rng_key)
train_loader = list(zip(X_train, y_train, strict=False))


# Create the model using Flax's linen module
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(50)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


# Initialize the model
def create_model(rng):
    model = MLP()
    params = model.init(rng, jnp.ones([1, 1]))
    return model, params


# Mean squared error loss function
def mse_loss(params, model, X, y):
    predictions = model.apply(params, X)
    loss = jnp.mean((predictions - y) ** 2)
    return loss


# Update function using Optax
def update(params, opt_state, X, y, model, optimizer):
    loss, grads = value_and_grad(mse_loss)(params, model, X, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


# Training loop
# Assuming `train_loader` is a generator that yields (X, y) batches

def init_model(rng_key):
    rng, init_rng = random.split(rng_key)
    model, params = create_model(init_rng)
    return model, params


def train_model(train_loader, n_epochs, model, params):
    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(params)

    for _ in range(n_epochs):
        for X, y in train_loader:
            params, opt_state, _ = update(params, opt_state, X, y, model, optimizer)
            # You can log the loss here if you want
    return params

n_epochs = 10
rng_key = random.key(711)
model, params = init_model(rng_key)
#final_params = train_model(train_loader, n_epochs, model, params)

with open('model_params.pkl', 'rb') as f:
    final_params = pickle.load(f)

loss_fn = lambda y_pred, y_true: jnp.mean((y_pred - y_true) ** 2) # noqa: E731
model_fn = lambda p, x: model.apply(p, x)  # noqa: E731, PLW0108
full_fn = lambda p, data: loss_fn(model_fn(p, data[0]), data[1])

hessian_dense = todense(lambda v: hvp(full_fn, final_params, (X_train, y_train), v), len([p for param in jax.tree.flatten(final_params)[0] for p in param]))
flat_hessian_new = flatten_hessian(hessian_dense, final_params)
#TODO: training speichern

assert jnp.allclose(hessian_dense, flat_hessian_new)