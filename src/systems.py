
from __future__ import annotations
from typing import Callable, Tuple, Optional, Literal
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from jax import random as jrandom

Array = jax.Array

# -----------------------------
# Utilities
# -----------------------------

def _make_tgrid(n_total_steps: int, step: float) -> Array:
    """Build a uniform time grid [0, step, 2*step, ..., horizon]. (horizon = n_total_steps * step)"""
    # ensure last point is exactly horizon if divisible, else n*step
    t = jnp.arange(n_total_steps + 1) * step
    return t


def _rk4_step(rhs: Callable[[Array], Array], x: Array, t: Array, h: float) -> Array:
    """One RK4 step for x' = rhs(t, x)."""
    # k1 = rhs(t, x)
    # k2 = rhs(t + 0.5 * h, x + 0.5 * h * k1)
    # k3 = rhs(t + 0.5 * h, x + 0.5 * h * k2)
    # k4 = rhs(t + h,       x + h * k3)
    k1 = rhs(x)
    k2 = rhs(x + 0.5 * h * k1)
    k3 = rhs(x + 0.5 * h * k2)
    k4 = rhs(x + h * k3)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _grid_points_in_box(lo: Array, hi: Array, num: int) -> Array:
    """Generate ~num grid points in a D-dim box by taking K^D lattice and truncating to num.

    Args:
      lo, hi: shape (D,)
      num: desired number of points (>=1)

    Returns:
      pts: shape (num, D)
    """
    D = lo.shape[0]
    # choose K so that K^D >= num
    K = int(jnp.ceil(num ** (1.0 / max(D, 1))))
    # 1D linspaces for each dim
    axes = [jnp.linspace(lo[i], hi[i], K) for i in range(D)]
    # meshgrid -> (D, K,...,K) then reshape to (K^D, D)
    mesh = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, D)
    # take first `num`
    return mesh[:num, :]


def _random_points_in_box(key: Array, lo: Array, hi: Array, num: int) -> Array:
    """Uniform random points in a D-dim box. Returns shape (num, D)."""
    D = lo.shape[0]
    u = jrandom.uniform(key, (num, D))
    return lo + u * (hi - lo)


# -----------------------------
# 1) Dynamics system
# -----------------------------

class CT_Dyn_Sys:
    """Dynamics system x' = f(x) with trajectory sampling utilities.
    rhs: callable(x)->dx with shape: x:(D,), returns dx:(D,)
    state_dim: dimension D
    """
    def __init__(self,
                 rhs: Callable[[Array], Array],
                 state_dim: int,
                 step_size: float): # step size for internal RK4
        self.rhs = rhs
        self.state_dim = state_dim
        self.step_size = step_size

    # ---- single-IC simulation ----
    def _simulate(self, x0: Array, start: float, n_steps: int, solver: Literal["rk4"]="rk4") -> Array:
        """Simulate one trajectory along t_grid.

        Args:
          x0: (D,)
          t_grid: (T,) ascending times
        Returns:
          traj: (T, D) with traj[0]=x0
        """
        assert solver == "rk4", "Only rk4 is implemented for now."
        D = self.state_dim
        assert x0.shape == (D,), f"x0 must have shape ({D},)"

        def one_step(carry, _):
            t_prev, x_prev = carry
            x_new = _rk4_step(self.rhs, x_prev, t_prev, self.step_size)
            return (t_prev + self.step_size, x_new), x_new

        # carry initial (t0=0 based on t_grid) and collect x's
        init = (start, x0)
        (_, _), xs = lax.scan(one_step, init,  length=n_steps)
        traj = jnp.vstack([x0[None, :], xs])
        return traj

    # ---- batched sampling ----
    def simulate(
        self,
        X0_lo: Array,
        X0_hi: Array,
        n_total_steps: int,
        *,
        n_samples: int,
        sampler: Literal["random","grid"]="random",
        key: Array=jrandom.PRNGKey(0),
        solver: Literal["rk4"]="rk4",
    ) -> Array:
        """Sample trajectories for a continuous-time autonomous system.

        Args:
          X0_lo, X0_hi: Arrays of shape `(state_dim,)` giving the initial box.
          n_total_steps: Number of integration steps.
          n_samples: Number of initial states to sample from the box.
          sampler: Sampling strategy for the initial states.
          key: PRNG key used when `sampler="random"`.
          solver: Time integrator to use. Only `"rk4"` is currently supported.

        Returns:
          A pair `(t_grid, trajs)` where `t_grid` has shape `(n_total_steps + 1,)`
          and `trajs` has shape `(n_samples, n_total_steps + 1, state_dim)`.
        """
        assert solver == "rk4", "Only rk4 is implemented for now."
        D = self.state_dim
        assert X0_lo.shape == (D,) and X0_hi.shape == (D,), "bounds must be (D,)"
        if sampler == "grid":
            X0s = _grid_points_in_box(X0_lo, X0_hi, n_samples)  # (B,D)
        else:
            X0s = _random_points_in_box(key, X0_lo, X0_hi, n_samples)

        t_grid = _make_tgrid(n_total_steps, self.step_size)
        sim_one = lambda x0: self._simulate(x0, start=0, n_steps=n_total_steps, solver=solver)
        trajs = jax.vmap(sim_one)(X0s)  # (B,T,D)
        return t_grid, trajs


# -----------------------------
# 2) NN-controlled ODE system (ZOH)
# -----------------------------

class CT_Ctl_Sys:
    """Closed-loop system with augmented state z=[x_state, u_action].
    """
    def __init__(self,
                 rhs: Callable[[Array], Array],
                 state_dim: int,
                 action_dim: int,
                 controller: Callable[[Array], Array],
                 n_steps_per_control: int,
                 step_size: float,
                 reference_dim: int = 0):
        self.rhs = rhs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.controller = controller
        self.n_steps_per_control = n_steps_per_control
        self.step_size = step_size
        self.reference_dim = reference_dim

        self.total_dim = self.state_dim + self.action_dim
        # Internal ODE engine over the augmented state
        self.dynamics = CT_Dyn_Sys(rhs=self.rhs, state_dim=self.total_dim, step_size=self.step_size)

    def _normalize_reference_seq(
        self,
        reference_seq: Optional[Array],
        *,
        n_control_steps: int,
        dtype,
    ) -> Array:
        if reference_seq is None:
            return jnp.zeros((n_control_steps, self.reference_dim), dtype=dtype)

        reference_seq = jnp.asarray(reference_seq, dtype=dtype)
        expected_shape = (n_control_steps, self.reference_dim)
        if reference_seq.shape != expected_shape:
            raise ValueError(
                "reference_seq must have shape "
                f"{expected_shape}, got {reference_seq.shape}."
            )
        return reference_seq

    def _apply_controller(self, x_aug: Array, reference: Optional[Array] = None) -> Array:
        """Overwrite action coordinates with controller output at a boundary.
        Args:
          x_aug: (D,), augmented [x_state, u_action]
        Returns:
          x_aug': (D,), with last Da entries set to controller(x_state).
        """
        x_s = x_aug[:self.state_dim]
        if reference is None:
            u = self.controller(x_s)  # (Da,)
        else:
            u = self.controller(jnp.concatenate([x_s, reference], axis=0))  # (Da,)
        return jnp.concatenate([x_s, u], axis=0)

    def _simulate(self, z0: Array, *, start: float, n_control_steps: int, reference_seq: Optional[Array] = None) -> Array:
        """Simulate one closed-loop trajectory; returns (n_control_steps+1, D)."""
        reference_seq = self._normalize_reference_seq(
            reference_seq,
            n_control_steps=n_control_steps,
            dtype=z0.dtype,
        )

        def one_interval(carry, xs):
            t_curr, z_curr = carry
            reference = xs
            # ZOH at boundary
            z_curr = self._apply_controller(z_curr, reference)
            # Integrate K steps with CT_Dyn_Sys (augmented RHS encodes du=0)
            seg = self.dynamics._simulate(z_curr, start=t_curr, n_steps=self.n_steps_per_control)  # (K+1, D)
            z_next = seg[-1]
            t_next = t_curr + self.n_steps_per_control * self.step_size
            seg_wo_first = seg[1:]  # (K,D)
            return (t_next, z_next), seg_wo_first

        (_, _), segs = lax.scan(one_interval, (start, z0), reference_seq, length=n_control_steps)  # segs:(M,K,D)
        z0_applied = self._apply_controller(z0, reference=reference_seq[0] if reference_seq is not None else None)[None, :]                    # (1,D)
        traj = jnp.concatenate([z0_applied, segs.reshape(-1, self.total_dim)], axis=0)
        return traj

    def simulate(
        self,
        Z0_lo: Array,
        Z0_hi: Array,
        n_total_steps: int,
        *,
        n_samples: int,
        sampler: Literal["random","grid"]="random",
        key: Array=jrandom.PRNGKey(0),
        reference_seq: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Sample continuous-time closed-loop trajectories from an augmented box.

        Args:
          Z0_lo, Z0_hi: Arrays of shape `(state_dim + action_dim,)` giving the
            initial augmented-state box.
          n_total_steps: Number of integration steps.
          n_samples: Number of initial states to sample from the box.
          sampler: Sampling strategy for the initial states.
          key: PRNG key used when `sampler="random"`.
          reference_seq: Optional shared reference sequence of shape
            `(n_control_steps, reference_dim)`. If omitted, a zero reference is used.

        Returns:
          A pair `(t_grid, trajs)` where `t_grid` has shape `(n_total_steps + 1,)`
          and `trajs` has shape
          `(n_samples, n_total_steps + 1, state_dim + action_dim)`.
        """
        if sampler == "grid":
            Z0s = _grid_points_in_box(Z0_lo, Z0_hi, n_samples)
        else:
            Z0s = _random_points_in_box(key, Z0_lo, Z0_hi, n_samples)

        t_grid = _make_tgrid(n_total_steps, self.step_size)
        n_control_steps = round(n_total_steps / self.n_steps_per_control)
        
        sim_one = lambda z0: self._simulate(z0, start=0.0, n_control_steps=n_control_steps, reference_seq=reference_seq)
        trajs = jax.vmap(sim_one)(Z0s)
        return t_grid, trajs


class CT_Plan_Sys:
    """Open-loop planning system with augmented state z=[x_state, u_action].
    """
    def __init__(self,
                 rhs: Callable[[Array], Array],
                 state_dim: int,
                 action_dim: int,
                 n_steps_per_plan: int,
                 step_size: float):
        self.rhs = rhs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_steps_per_plan = n_steps_per_plan
        self.step_size = step_size

        self.total_dim = self.state_dim + self.action_dim
        # Internal dynamics engine over the augmented state
        self.dynamics = CT_Dyn_Sys(rhs=self.rhs, state_dim=self.total_dim, step_size=self.step_size)

    def _normalize_action_seq(
        self,
        action_seq: Array,
        *,
        n_total_steps: int,
        dtype,
    ) -> Array:
        action_seq = jnp.asarray(action_seq, dtype=dtype)
        if action_seq.ndim != 3 or action_seq.shape[2] != self.action_dim:
            raise ValueError(
                "action_seq must have shape "
                f"(n_plan, n_plan_steps, {self.action_dim}), got {action_seq.shape}."
            )

        n_plan_steps = action_seq.shape[1]
        if n_total_steps != n_plan_steps * self.n_steps_per_plan:
            raise ValueError(
                "action_seq shape mismatch: expected n_total_steps == "
                f"n_plan_steps * n_steps_per_plan, got {n_total_steps} != "
                f"{n_plan_steps} * {self.n_steps_per_plan}."
            )
        return action_seq

    def _apply_plan(self, x_aug: Array, u: Array) -> Array:
        """Overwrite action coordinates with controller output at a boundary.
        Args:
          x_aug: (n_plan, D), augmented [x_state, u_action]
        Returns:
          x_aug': (n_plan, D), with last Da entries set to controller(x_state).
        """
        x_s = x_aug[:, :self.state_dim]
        return jnp.concatenate([x_s, u], axis=1)

    def _simulate(self, z0: Array, action_seq: Array, *, start: int) -> Array:
        """Simulate one open-loop trajectory; returns (n_plan, n_plan_steps+1, D)."""
        # z0: (n_plan, D), action_seq: (n_plan, n_plan_steps, Da)
        def one_interval(carry, u):
            t_curr, z_curr = carry
            # ZOH at boundary
            z_curr = self._apply_plan(z_curr, u)
            # Integrate K steps with CT_Dyn_Sys (augmented RHS encodes du=0)
            sim_one = lambda x0: self.dynamics._simulate(x0, start=t_curr, n_steps=self.n_steps_per_plan)
            seg = jax.vmap(sim_one)(z_curr)  # (n_plan, K+1, D)
            z_next = seg[:, -1]
            t_next = t_curr + self.n_steps_per_plan
            seg_wo_first = seg[:, 1:]  # (n_plan, K,D)
            return (t_next, z_next), seg_wo_first

        action_seq = jnp.transpose(action_seq, (1,0,2))  # (n_plan_steps, n_plan, Da)
        (_, _), segs = lax.scan(one_interval, (start, z0), action_seq)  # segs: (n_plan_steps, n_plan, K,D)
        
        z0_applied = self._apply_plan(z0, action_seq[0])[:,None,:]  # (n_plan,1,D)
        traj = jnp.concatenate([z0_applied, segs.reshape(z0_applied.shape[0], -1, self.total_dim)], axis=1)
        return traj

    def simulate(
        self,
        Z0_lo: Array,
        Z0_hi: Array,
        n_total_steps: int,
        action_seq: Array,
        *,
        n_samples: int,
        sampler: Literal["random","grid"]="random",
        key: Array=jrandom.PRNGKey(0),
    ) -> Tuple[Array, Array]:
        """Sample continuous-time open-loop trajectories for one or more plans.

        Args:
          Z0_lo, Z0_hi: Arrays of shape `(state_dim + action_dim,)` giving the
            initial augmented-state box.
          n_total_steps: Total number of integration steps.
          action_seq: Shared action plans with shape
            `(n_plan, n_plan_steps, action_dim)`, where
            `n_total_steps == n_plan_steps * n_steps_per_plan`.
          n_samples: Number of initial states to sample from the box.
          sampler: Sampling strategy for the initial states.
          key: PRNG key used when `sampler="random"`.

        Returns:
          A pair `(t_grid, trajs)` where `t_grid` has shape `(n_total_steps + 1,)`
          and `trajs` has shape
          `(n_samples, n_plan, n_total_steps + 1, state_dim + action_dim)`.
        """
        action_seq = self._normalize_action_seq(
            action_seq,
            n_total_steps=n_total_steps,
            dtype=Z0_lo.dtype,
        )

        if sampler == "grid":
            Z0s = _grid_points_in_box(Z0_lo, Z0_hi, n_samples)
        else:
            Z0s = _random_points_in_box(key, Z0_lo, Z0_hi, n_samples)

        n_plan, n_plan_steps = action_seq.shape[0], action_seq.shape[1]
        t_grid = _make_tgrid(n_total_steps, self.step_size)

        Z0s = Z0s[:, None, :].repeat(n_plan, axis=1)  # (n_samples, n_plan, D)
        
        sim_one = lambda z0: self._simulate(z0, action_seq, start=0)
        trajs = jax.vmap(sim_one)(Z0s)
        # trajs: (n_samples, n_plan, n_plan_steps+1, D)
        return t_grid, trajs



# -----------------------------
# 3) Discrete-time dynamics system
# -----------------------------

class DT_Dyn_Sys:
    """Discrete-time dynamics system x_{t+1} = f(x_t) with trajectory sampling utilities.
    rhs: callable(x_t)->x_{t+1} with shape: x_t:(D,), returns x_{t+1}:(D,)
    state_dim: dimension D
    """
    def __init__(self,
                 rhs: Callable[[Array], Array],
                 state_dim: int,
                 step_size: int = 1):
        self.rhs = rhs
        self.state_dim = state_dim
        self.step_size = step_size
        assert step_size == 1, "For DT_Dyn_Sys, step_size must be 1."

    # ---- single-IC simulation ----
    def _simulate(self, x0: Array, start: int, n_steps: int) -> Array:
        """Simulate one trajectory with n_steps discrete steps.

        Args:
          x0: (D,)
          n_steps: number of discrete steps
        Returns:
          traj: (T, D) with traj[0]=x0
        """
        D = self.state_dim
        assert x0.shape == (D,), f"x0 must have shape ({D},)"

        def one_step(carry, _):
            t_prev, x_prev = carry
            x_new = self.rhs(x_prev)
            return (t_prev + self.step_size, x_new), x_new

        # carry initial (t0=0 based on t_grid) and collect x's
        init = (start, x0)
        (_, _), xs = lax.scan(one_step, init,  length=n_steps)
        traj = jnp.vstack([x0[None, :], xs])
        return traj

    # ---- batched sampling ----
    def simulate(
        self,
        X0_lo: Array,
        X0_hi: Array,
        n_total_steps: int,
        *,
        n_samples: int,
        sampler: Literal["random","grid"]="random",
        key: Array=jrandom.PRNGKey(0),
    ) -> Array:
        """Sample trajectories for a discrete-time autonomous system.

        Args:
          X0_lo, X0_hi: Arrays of shape `(state_dim,)` giving the initial box.
          n_total_steps: Number of discrete-time steps.
          n_samples: Number of initial states to sample from the box.
          sampler: Sampling strategy for the initial states.
          key: PRNG key used when `sampler="random"`.

        Returns:
          A pair `(t_grid, trajs)` where `t_grid` has shape `(n_total_steps + 1,)`
          and `trajs` has shape `(n_samples, n_total_steps + 1, state_dim)`.
        """
        D = self.state_dim
        assert X0_lo.shape == (D,) and X0_hi.shape == (D,), "bounds must be (D,)"
        if sampler == "grid":
            X0s = _grid_points_in_box(X0_lo, X0_hi, n_samples)  # (B,D)
        else:
            X0s = _random_points_in_box(key, X0_lo, X0_hi, n_samples)

        t_grid = _make_tgrid(n_total_steps, self.step_size)
        sim_one = lambda x0: self._simulate(x0, start=0, n_steps=n_total_steps)
        trajs = jax.vmap(sim_one)(X0s)  # (B,T,D)
        return t_grid, trajs


# -----------------------------
# 4) NN-controlled discrete-time system (ZOH)
# -----------------------------

class DT_Ctl_Sys:
    """Closed-loop system with augmented state z=[x_state, u_action].
    """
    def __init__(self,
                 rhs: Callable[[Array], Array],
                 state_dim: int,
                 action_dim: int,
                 controller: Callable[[Array], Array],
                 n_steps_per_control: int,
                 step_size: int,
                 reference_dim: int = 0):
        self.rhs = rhs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.controller = controller
        self.n_steps_per_control = n_steps_per_control
        self.step_size = step_size
        self.reference_dim = reference_dim
        assert step_size == 1, "For DT_Dyn_Sys, step_size must be 1."

        self.total_dim = self.state_dim + self.action_dim
        # Internal dynamics engine over the augmented state
        self.dynamics = DT_Dyn_Sys(rhs=self.rhs, state_dim=self.total_dim, step_size=self.step_size)

    def _normalize_reference_seq(
        self,
        reference_seq: Optional[Array],
        *,
        n_control_steps: int,
        dtype,
    ) -> Array:
        if reference_seq is None:
            return jnp.zeros((n_control_steps, self.reference_dim), dtype=dtype)

        reference_seq = jnp.asarray(reference_seq, dtype=dtype)
        expected_shape = (n_control_steps, self.reference_dim)
        if reference_seq.shape != expected_shape:
            raise ValueError(
                "reference_seq must have shape "
                f"{expected_shape}, got {reference_seq.shape}."
            )
        return reference_seq

    def _apply_controller(self, x_aug: Array, reference: Optional[Array] = None) -> Array:
        """Overwrite action coordinates with controller output at a boundary.
        Args:
          x_aug: (D,), augmented [x_state, u_action]
        Returns:
          x_aug': (D,), with last Da entries set to controller(x_state).
        """
        x_s = x_aug[:self.state_dim]
        if reference is None:
            u = self.controller(x_s)  # (Da,)
        else:
            u = self.controller(jnp.concatenate([x_s, reference], axis=0))  # (Da,)
        return jnp.concatenate([x_s, u], axis=0)

    def _simulate(self, z0: Array, *, start: int, n_control_steps: int, reference_seq: Optional[Array] = None) -> Array:
        """Simulate one closed-loop trajectory; returns (n_control_steps+1, D)."""
        reference_seq = self._normalize_reference_seq(
            reference_seq,
            n_control_steps=n_control_steps,
            dtype=z0.dtype,
        )

        def one_interval(carry, xs):
            t_curr, z_curr = carry
            reference = xs
            # ZOH at boundary
            z_curr = self._apply_controller(z_curr, reference)
            # Integrate K steps with CT_Dyn_Sys (augmented RHS encodes du=0)
            seg = self.dynamics._simulate(z_curr, start=t_curr, n_steps=self.n_steps_per_control)  # (K+1, D)
            z_next = seg[-1]
            t_next = t_curr + self.n_steps_per_control
            seg_wo_first = seg[1:]  # (K,D)
            return (t_next, z_next), seg_wo_first

        (_, _), segs = lax.scan(one_interval, (start, z0), reference_seq, length=n_control_steps)  # segs:(M,K,D)
        z0_applied = self._apply_controller(z0, reference=reference_seq[0] if reference_seq is not None else None)[None, :]                    # (1,D)
        traj = jnp.concatenate([z0_applied, segs.reshape(-1, self.total_dim)], axis=0)
        return traj

    def simulate(
        self,
        Z0_lo: Array,
        Z0_hi: Array,
        n_total_steps: int,
        *,
        n_samples: int,
        sampler: Literal["random","grid"]="random",
        key: Array=jrandom.PRNGKey(0),
        reference_seq: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Sample discrete-time closed-loop trajectories from an augmented box.

        Args:
          Z0_lo, Z0_hi: Arrays of shape `(state_dim + action_dim,)` giving the
            initial augmented-state box.
          n_total_steps: Number of discrete-time steps.
          n_samples: Number of initial states to sample from the box.
          sampler: Sampling strategy for the initial states.
          key: PRNG key used when `sampler="random"`.
          reference_seq: Optional shared reference sequence of shape
            `(n_control_steps, reference_dim)`. If omitted, a zero reference is used.

        Returns:
          A pair `(t_grid, trajs)` where `t_grid` has shape `(n_total_steps + 1,)`
          and `trajs` has shape
          `(n_samples, n_total_steps + 1, state_dim + action_dim)`.
        """
        if sampler == "grid":
            Z0s = _grid_points_in_box(Z0_lo, Z0_hi, n_samples)
        else:
            Z0s = _random_points_in_box(key, Z0_lo, Z0_hi, n_samples)

        t_grid = _make_tgrid(n_total_steps, self.step_size)
        n_control_steps = round(n_total_steps / self.n_steps_per_control)
        sim_one = lambda z0: self._simulate(z0, start=0, n_control_steps=n_control_steps, reference_seq=reference_seq)
        trajs = jax.vmap(sim_one)(Z0s)
        return t_grid, trajs


class DT_Plan_Sys:
    """Open-loop planning system with augmented state z=[x_state, u_action].
    """
    def __init__(self,
                 rhs: Callable[[Array], Array],
                 state_dim: int,
                 action_dim: int,
                 n_steps_per_plan: int,
                 step_size: int):
        self.rhs = rhs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_steps_per_plan = n_steps_per_plan
        self.step_size = step_size
        assert step_size == 1, "For DT_Dyn_Sys, step_size must be 1."

        self.total_dim = self.state_dim + self.action_dim
        # Internal dynamics engine over the augmented state
        self.dynamics = DT_Dyn_Sys(rhs=self.rhs, state_dim=self.total_dim, step_size=self.step_size)

    def _normalize_action_seq(
        self,
        action_seq: Array,
        *,
        n_total_steps: int,
        dtype,
    ) -> Array:
        action_seq = jnp.asarray(action_seq, dtype=dtype)
        if action_seq.ndim != 3 or action_seq.shape[2] != self.action_dim:
            raise ValueError(
                "action_seq must have shape "
                f"(n_plan, n_plan_steps, {self.action_dim}), got {action_seq.shape}."
            )

        n_plan_steps = action_seq.shape[1]
        if n_total_steps != n_plan_steps * self.n_steps_per_plan:
            raise ValueError(
                "action_seq shape mismatch: expected n_total_steps == "
                f"n_plan_steps * n_steps_per_plan, got {n_total_steps} != "
                f"{n_plan_steps} * {self.n_steps_per_plan}."
            )
        return action_seq

    def _apply_plan(self, x_aug: Array, u: Array) -> Array:
        """Overwrite action coordinates with controller output at a boundary.
        Args:
          x_aug: (n_plan, D), augmented [x_state, u_action]
        Returns:
          x_aug': (n_plan, D), with last Da entries set to controller(x_state).
        """
        x_s = x_aug[:, :self.state_dim]
        return jnp.concatenate([x_s, u], axis=1)

    def _simulate(self, z0: Array, action_seq: Array, *, start: int) -> Array:
        """Simulate one open-loop trajectory; returns (n_plan, n_plan_steps+1, D)."""
        # z0: (n_plan, D), action_seq: (n_plan, n_plan_steps, Da)
        def one_interval(carry, u):
            t_curr, z_curr = carry
            # ZOH at boundary
            z_curr = self._apply_plan(z_curr, u)
            # Integrate K steps with CT_Dyn_Sys (augmented RHS encodes du=0)
            sim_one = lambda x0: self.dynamics._simulate(x0, start=t_curr, n_steps=self.n_steps_per_plan)
            seg = jax.vmap(sim_one)(z_curr)  # (n_plan, K+1, D)
            z_next = seg[:, -1]
            t_next = t_curr + self.n_steps_per_plan
            seg_wo_first = seg[:, 1:]  # (n_plan, K,D)
            return (t_next, z_next), seg_wo_first

        action_seq = jnp.transpose(action_seq, (1,0,2))  # (n_plan_steps, n_plan, Da)
        (_, _), segs = lax.scan(one_interval, (start, z0), action_seq)  # segs: (n_plan_steps, n_plan, K,D)
        
        z0_applied = self._apply_plan(z0, action_seq[0])[:,None,:]  # (n_plan,1,D)
        traj = jnp.concatenate([z0_applied, segs.reshape(z0_applied.shape[0], -1, self.total_dim)], axis=1)
        return traj

    def simulate(
        self,
        Z0_lo: Array,
        Z0_hi: Array,
        n_total_steps: int,
        action_seq: Array,
        *,
        n_samples: int,
        sampler: Literal["random","grid"]="random",
        key: Array=jrandom.PRNGKey(0),
    ) -> Tuple[Array, Array]:
        """Sample discrete-time open-loop trajectories for one or more plans.

        Args:
          Z0_lo, Z0_hi: Arrays of shape `(state_dim + action_dim,)` giving the
            initial augmented-state box.
          n_total_steps: Total number of discrete-time steps.
          action_seq: Shared action plans with shape
            `(n_plan, n_plan_steps, action_dim)`, where
            `n_total_steps == n_plan_steps * n_steps_per_plan`.
          n_samples: Number of initial states to sample from the box.
          sampler: Sampling strategy for the initial states.
          key: PRNG key used when `sampler="random"`.

        Returns:
          A pair `(t_grid, trajs)` where `t_grid` has shape `(n_total_steps + 1,)`
          and `trajs` has shape
          `(n_samples, n_plan, n_total_steps + 1, state_dim + action_dim)`.
        """
        action_seq = self._normalize_action_seq(
            action_seq,
            n_total_steps=n_total_steps,
            dtype=Z0_lo.dtype,
        )

        if sampler == "grid":
            Z0s = _grid_points_in_box(Z0_lo, Z0_hi, n_samples)
        else:
            Z0s = _random_points_in_box(key, Z0_lo, Z0_hi, n_samples)

        n_plan, n_plan_steps = action_seq.shape[0], action_seq.shape[1]
        t_grid = _make_tgrid(n_total_steps, self.step_size)

        Z0s = Z0s[:, None, :].repeat(n_plan, axis=1)  # (n_samples, n_plan, D)
        
        sim_one = lambda z0: self._simulate(z0, action_seq, start=0)
        trajs = jax.vmap(sim_one)(Z0s)
        # trajs: (n_samples, n_plan, n_plan_steps+1, D)
        return t_grid, trajs
