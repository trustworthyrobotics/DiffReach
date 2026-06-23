import jax.numpy as jnp

"""
dynamics_expressions:
  - "x2"
  - "(1.0 - x1 * x1) * x2 - x1"
"""

def dynamics(x):
    x1, x2 = x[0], x[1]
    return jnp.stack([x2, (1.0 - x1 * x1) * x2 - x1], axis=0)
