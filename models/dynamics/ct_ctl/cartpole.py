import jax.numpy as jnp

"""
dynamics_expressions:
  - "x2"
  - "2 * u1"
  - "x4"
  - "(0.08*0.41*(9.8 * sin(x3) - 2*u1 * cos(x3)) - 0.0021 * x4) / 0.0105"
  - "0"
"""

def dynamics(x):
    x1,x2,x3,x4,u1 = [x[i] for i in range(5)]
    dx1 = x2
    dx2 = 2 * u1
    dx3 = x4
    dx4 = (0.08*0.41*(9.8 * jnp.sin(x3) - 2*u1 * jnp.cos(x3)) - 0.0021 * x4) / 0.0105
    return jnp.stack([dx1, dx2, dx3, dx4], axis=0)