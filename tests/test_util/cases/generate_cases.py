import jax
import random
from jax import numpy as jnp


# Helper-Funktion zur dynamischen Generierung von Pytrees
def generate_random_pytree(depth=2, max_branches=3, seed=1):
    if depth == 0:
        return jnp.array(random.randint(0, 10))
    pytree_type = random.choice(["dict", "list", "tuple"])
    if pytree_type == "dict":
        return {f"key_{i}": generate_random_pytree(depth - 1) for i in range(random.randint(1, max_branches))}
    elif pytree_type == "list":
        return [generate_random_pytree(depth - 1) for _ in range(random.randint(1, max_branches))]
    elif pytree_type == "tuple":
        return tuple(generate_random_pytree(depth - 1) for _ in range(random.randint(1, max_branches)))


a = generate_random_pytree(3, 8)
