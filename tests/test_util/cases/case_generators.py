import random
import jax
import jax.numpy as jnp
from laplax.util.flatten import create_pytree_flattener


def generate_random_pytree(depth=2, max_branches=3, seed=None):
    if seed is not None:
        random.seed(seed)
    if depth == 0:
        return jax.random.normal(jax.random.PRNGKey(random.randint(0, 1000)), shape=(random.randint(1, 5),))
    pytree_type = random.choice(["dict", "list", "tuple"])
    if pytree_type == "dict":
        return {f"key_{i}": generate_random_pytree(depth - 1, max_branches) for i in range(random.randint(1, max_branches))}
    elif pytree_type == "list":
        return [generate_random_pytree(depth - 1, max_branches) for _ in range(random.randint(1, max_branches))]
    elif pytree_type == "tuple":
        return tuple(generate_random_pytree(depth - 1, max_branches) for _ in range(random.randint(1, max_branches)))


def case_random_pytree():
    for seed in [1, 42, 256]:
        for depth in [1, 2, 3]:
            for max_branches in [2, 3, 4]:
                pytree = generate_random_pytree(depth=depth, max_branches=max_branches, seed=seed)
                flatten, _ = create_pytree_flattener(pytree)
                vector = flatten(pytree)
                yield pytree, vector


def case_two_pytree():
    for seed in [1, 42, 256]:
        for depth in [1, 2, 3]:
            for max_branches in [2, 3, 4]:
                pytree1 = generate_random_pytree(depth=depth, max_branches=max_branches, seed=seed)
                pytree2 = generate_random_pytree(depth=depth, max_branches=max_branches, seed=seed)
                flatten1, _ = create_pytree_flattener(pytree1)
                flatten2, _ = create_pytree_flattener(pytree2)
                vector1 = flatten1(pytree1)
                vector2 = flatten2(pytree2)

                yield (pytree1, vector1), (pytree2, vector2)