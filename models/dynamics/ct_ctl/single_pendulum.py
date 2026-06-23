import jax.numpy as jnp 

"""
dynamics_expressions:
  - "x2"
  - "2 * sin(x1) + 8 * u1"
  - "0"
"""

def dynamics(x):
    x1,x2,u1 = [x[i] for i in range(3)]
    dx1 = x2
    dx2 = 2 * jnp.sin(x1) + 8 * u1
    return jnp.stack([dx1, dx2], axis=0)
