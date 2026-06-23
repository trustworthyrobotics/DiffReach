import jax.numpy as jnp

"""
dynamics_expressions:
  - "cos(x8)*cos(x9)*x4 + (sin(x7)*sin(x8)*cos(x9) - cos(x7)*sin(x9))*x5 + (cos(x7)*sin(x8)*cos(x9) + sin(x7)*sin(x9))*x6"
  - "cos(x8)*sin(x9)*x4 + (sin(x7)*sin(x8)*sin(x9) + cos(x7)*cos(x9))*x5 + (cos(x7)*sin(x8)*sin(x9) - sin(x7)*cos(x9))*x6"
  - "sin(x8)*x4 - sin(x7)*cos(x8)*x5 - cos(x7)*cos(x8)*x6"
  - "x12*x5 - x11*x6 - 9.81*sin(x8)"
  - "x10*x6 - x12*x4 + 9.81*cos(x8)*sin(x7)"
  - "x11*x4 - x10*x5 + 9.81*cos(x8)*cos(x7) - 9.81 + 7.14285714285714*(x3 - 1) - 2.14285714285714*x6"
  - "x10 + (sin(x7)*(sin(x8)/cos(x8)))*x11 + (cos(x7)*(sin(x8)/cos(x8)))*x12"
  - "cos(x7)*x11 - sin(x7)*x12"
  - "(sin(x7)/cos(x8))*x11 + (cos(x7)/cos(x8))*x12"
  - "-0.92592592592593*x11*x12 - 18.51851851851852*(x7 + x10)"
  - "0.92592592592593*x10*x12 - 18.51851851851852*(x8 + x11)"
  - "0"
"""

def dynamics(x):
    # unpack for readability (use 1-based names from description)
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12 = [x[i] for i in range(12)]

    # precompute trig
    c7, s7 = jnp.cos(x7), jnp.sin(x7)   # roll
    c8, s8 = jnp.cos(x8), jnp.sin(x8)   # pitch
    c9, s9 = jnp.cos(x9), jnp.sin(x9)   # yaw

    tan8 = jnp.tan(x8)
    g = 9.81

    # Position kinematics from body-frame velocities (x4,x5,x6)
    dx1 =  c8 * c9 * x4 + (s7 * s8 * c9 - c7 * s9) * x5 + (c7 * s8 * c9 + s7 * s9) * x6
    dx2 =  c8 * s9 * x4 + (s7 * s8 * s9 + c7 * c9) * x5 + (c7 * s8 * s9 - s7 * c9) * x6
    dx3 =  s8 * x4     -  s7 * c8 * x5                -  c7 * c8 * x6

    # Body linear accelerations with gravity and couplings
    dx4 =  x12 * x5 - x11 * x6 - g * s8
    dx5 =  x10 * x6 - x12 * x4 + g * c8 * s7
    dx6 =  x11 * x4 - x10 * x5 + g * c8 * c7 - g \
           + 7.14285714285714 * (x3 - 1.0) - 2.14285714285714 * x6

    # Euler angle rates (standard ZYX convention)
    dx7 = x10 + s7 * tan8 * x11 + c7 * tan8 * x12
    dx8 = c7 * x11 - s7 * x12
    dx9 = (s7 / c8) * x11 + (c7 / c8) * x12

    # Simple first-order body-rate dynamics with cross-terms and damping
    dx10 = -0.92592592592593 * x11 * x12 - 18.51851851851852 * (x7 + x10)
    dx11 =  0.92592592592593 * x10 * x12 - 18.51851851851852 * (x8 + x11)
    dx12 =  jnp.zeros_like(x12)

    return jnp.stack([dx1,dx2,dx3,dx4,dx5,dx6,dx7,dx8,dx9,dx10,dx11,dx12], axis=0)
