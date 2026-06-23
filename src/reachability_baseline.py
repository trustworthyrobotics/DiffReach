
from __future__ import annotations
from typing import Callable, Tuple, Optional

import jax
import jax.numpy as jnp

import src.settings as settings
from src.taylor_model import Interval, QuadPoly, QuadTM
from src.crown_wrapper import crown

from jax_verify import IntervalBound, backward_crown_bound_propagation

from src.systems import CT_Dyn_Sys, CT_Ctl_Sys, CT_Plan_Sys, _make_tgrid, DT_Dyn_Sys, DT_Ctl_Sys, DT_Plan_Sys
from src.reachability import DT_Dyn_Reach, CT_Dyn_Reach

Array = jax.Array
import immrax as irx
from immutabledict import immutabledict

class DT_Plan_Reach_CROWN(DT_Plan_Sys):
    """Open-loop planned DT system reachability.

    rhs: Callable[[Array], Array]  # augmented-state RHS x' = f(t,x) with du=0

    Inherits CT_Ctl_Sys to reuse configuration and simulation API,
    and CONTAINS an CT_Dyn_Reach instance sized for the augmented state (D_s + D_a).
    The outer loop performs controller updates (via CROWN) and the inner loop
    advances K ODE steps by delegating to CT_Dyn_Reach.step_once().
    """
    def __init__(self,
                 rhs: Callable[[Array], Array],
                 state_dim: int,
                 action_dim: int,
                 nn_dyn: bool,   
                 n_steps_per_plan: int,
                 step_size: int,
                 sr_window_size: int = 100,
                 config: dict = None,
                 *args, **kwargs
                 ):
        assert step_size == 1, "For DT_Plan_Reach, step_size must be 1."
        super().__init__(rhs=rhs,
                         state_dim=state_dim,
                         action_dim=action_dim,
                         n_steps_per_plan=n_steps_per_plan,
                         step_size=step_size)
        self.nn_dyn = nn_dyn

        if config is not None:
            settings.update_config(config)

        self.Ds = int(self.state_dim)
        self.Da = int(self.action_dim)
        self.D  = self.Ds + self.Da
        self.V  = self.D + 1
        # Inner ODE reach core over augmented state
        self.dyn_reach = DT_Dyn_Reach(
            rhs=self.rhs,                # augmented-state RHS with du=0
            state_dim=self.D,
            nn_dyn=self.nn_dyn,
            step_size=self.step_size,
            sr_window_size=sr_window_size,
        )

    def make_plan(self, act_seq):
        # act_seq: (n_horizon, n_partition*n_plan, action_dim)
        horizon = act_seq.shape[0]
        def plan_fixed_action(z):
            # z: (n_partition*n_plan, state_dim+action_dim)
            x_curr = z[:, :self.Ds]  # (n_partition*n_plan, state_dim)
            def step(x_curr, u_t):
                inp = jnp.concatenate([x_curr, u_t], axis=-1)
                z_next = jax.vmap(self.rhs)(inp)
                x_next = z_next[:, :self.Ds]
                return x_next, z_next  # carry, y
            xT, zs_next = jax.lax.scan(step, x_curr, act_seq)          # xs_next: (horizon, n_partition*n_plan, state_dim)
            zs = jnp.concatenate([z[None, :], zs_next], 0)  # (horizon+1, n_partition*n_plan, state_dim)
            return zs  # (n_horizon+1, n_partition*n_plan, D)
        return plan_fixed_action

    def verify(self, Z0_lo: Array, Z0_hi: Array, n_total_steps: int, action_seq: Array):
        # Z0_lo, Z0_hi: (B, D)
        # action_seq: (B, n_plan, n_horizon, action_dim)
        n_plan = action_seq.shape[1]
        n_partition = Z0_lo.shape[0]
        action_seq = action_seq.astype(Z0_lo.dtype)
        Z0_lo = Z0_lo.repeat(n_plan, axis=0)    # (n_partition*n_plan, D)
        Z0_hi = Z0_hi.repeat(n_plan, axis=0)    # (n_partition*n_plan, D)
        action_seq = action_seq.reshape(n_plan*n_partition, -1, self.Da).transpose((1, 0, 2))  # (n_horizon, n_partition*n_plan, action_dim)

        B = Z0_lo.shape[0]
        D = self.D
        iv0 = Interval(Z0_lo, Z0_hi)

        plan_func = self.make_plan(action_seq)
        def call_crown(z_lo, z_hi):
            out = backward_crown_bound_propagation(plan_func, IntervalBound(z_lo, z_hi))
            return out.lower, out.upper

        los_all, his_all = call_crown(Z0_lo, Z0_hi)  # (n_horizon+1, n_partition*n_plan, D)

        # Stitch intervals
        lowers = los_all.reshape(-1, B, D).transpose((1, 0, 2))
        uppers = his_all.reshape(-1, B, D).transpose((1, 0, 2))
        
        lowers = lowers.reshape(n_partition, n_plan, -1, D)
        uppers = uppers.reshape(n_partition, n_plan, -1, D)
        times  = _make_tgrid(n_total_steps, self.step_size)
        return times, lowers, uppers, None, jnp.ones((n_total_steps, B, D))  # placeholder for init_shrinked_all

    def verify_w_model(self, rhs: Callable[[Array], Array], X0_lo: Array, X0_hi: Array, n_total_steps: int, action_seq: Array) -> Tuple[Array, Array, Array, Array]:
        """Verify with learned dynamics model (overrides self.rhs)."""
        self.rhs = rhs
        self.dyn_reach._set_rhs(rhs)
        return self.verify(X0_lo, X0_hi, n_total_steps, action_seq)

class CT_Dyn_Reach_immrax(CT_Dyn_Sys):
    """Refactored dynamics reachability with a class API and clear subroutines.

    The class organizes the original pipeline into named steps that mirror
    the math: seeding, parameterization, Picard updates, and interval logging.
    """
    def __init__(self,
                 rhs: Callable[[Array], Array],
                 state_dim: int,
                 nn_dyn: bool,
                 step_size: float,
                 # reach parameters (see flowpipe.py for details)
                 init_remainder: float = 1e-1,
                 frr_rounds: int = 3,
                 frr_stop_ratio: float = 0.95,
                 sr_window_size: int = 5,
                 config: dict = None,
                 ):
        super().__init__(rhs=rhs, state_dim=state_dim, step_size=step_size)

        if config is not None:
            settings.update_config(config)
        self.nn_dyn = nn_dyn

        self.D = self.state_dim
        self.V = self.D + 1

        olsys = OpenLoopSystem_dyn(state_dim=self.D, rhs=self.rhs)
        self.emb = irx.natemb(olsys)

        def compute_traj(x0, t_end):
            return self.emb.compute_trajectory(0.0, t_end, x0, (), self.step_size, solver="rk45")

        self._verify = jax.vmap(compute_traj, (0, None))

    # ---- public entrypoints ----
    def verify(self, X0_lo: Array, X0_hi: Array, n_total_steps: int) -> Tuple[Array, Array, Array, Array]:
        # Normalize, build boxes & initial models
        x0s = jnp.concatenate([X0_lo[:, :self.D], X0_hi[:, :self.D]], axis=1)  # (B, 2*D)

        t_end = n_total_steps * self.step_size
        trajs = self._verify(x0s, t_end)
        times = trajs.ts[0, :n_total_steps+1]
        ys = trajs.ys[:, :n_total_steps+1, :]  # (B, n_total_steps, D)

        lowers = ys[:, :, :self.D]
        uppers = ys[:, :, self.D:]

        return times, lowers, uppers, None, jnp.ones((n_total_steps, X0_lo.shape[0], self.D))  # placeholder for init_shrinked_all

class CT_Ctl_Reach_immrax(CT_Ctl_Sys):
    """Neural-network controlled system reachability.

    rhs: Callable[[Array], Array]  # augmented-state RHS x' = f(t,x) with du=0


    Inherits CT_Ctl_Sys to reuse configuration and simulation API,
    and CONTAINS an CT_Dyn_Reach instance sized for the augmented state (D_s + D_a).
    The outer loop performs controller updates (via CROWN) and the inner loop
    advances K ODE steps by delegating to CT_Dyn_Reach.step_once().
    """
    def __init__(self,
                 rhs: Callable[[Array], Array],
                 state_dim: int,
                 action_dim: int,
                 nn_dyn: bool,   
                 controller: Callable[[Array], Array],
                 n_steps_per_control: int,
                 step_size: float,
                 # reach parameters (see flowpipe.py for details)
                 init_remainder: float = 1e-1,
                 frr_rounds: int = 3,
                 frr_stop_ratio: float = 0.95,
                 sr_window_size: int = 100,
                 reference_dim: int = 0,
                 config: dict = None,
                 ):
        assert n_steps_per_control == 1, "For DT_Ctl_Reach_immrax, n_steps_per_control must be 1."
        super().__init__(rhs=rhs,
                         state_dim=state_dim,
                         action_dim=action_dim,
                         controller=controller,
                         n_steps_per_control=n_steps_per_control,
                         step_size=step_size,
                         reference_dim=reference_dim)
        self.nn_dyn = nn_dyn

        if config is not None:
            settings.update_config(config)

        self.Ds = int(self.state_dim)
        self.Da = int(self.action_dim)
        self.Dr = int(self.reference_dim)
        self.D  = self.Ds + self.Da
        self.V  = self.D + 1

        olsys = OpenLoopSystem(state_dim=self.Ds, rhs=self.rhs)
        clsys = irx.NNCSystem(olsys, controller, self.Da)
        self.clembsys = irx.NNCEmbeddingSystem(clsys, "crown", "local", "local")

        permutations = irx.standard_permutation(1 + self.Ds + self.Da + 1)
        corners = irx.two_corners(1 + self.Ds + self.Da + 1)
        w = irx.icentpert([0.0], 0.0)

        def w_map(t, x):
            return w

        def compute_traj(x0, t_end):
            return self.clembsys.compute_trajectory(0.0, t_end, x0, (w_map,), self.step_size, f_kwargs=immutabledict({"corners": corners, "permutations": permutations}), solver="euler")

        self._verify = jax.vmap(compute_traj, (0, None))

    def verify(self, Z0_lo: Array, Z0_hi: Array,
     n_total_steps: int, reference_seq: Optional[Array] = None) -> Tuple[Array, Array, Array, Array]:
        
        x0s = jnp.concatenate([Z0_lo[:, :self.Ds], Z0_hi[:, :self.Ds]], axis=1)  # (B, 2*Ds)

        t_end = n_total_steps * self.step_size
        trajs = self._verify(x0s, t_end)
        times = trajs.ts[0, :n_total_steps+1]
        ys = trajs.ys[:, :n_total_steps+1, :]  # (B, n_total_steps, Ds)

        lowers = ys[:, :, :self.Ds]
        uppers = ys[:, :, self.Ds:]

        return times, lowers, uppers, None, jnp.ones((n_total_steps, Z0_lo.shape[0], self.Ds))  # placeholder for init_shrinked_all

    def verify_w_model(self, rhs: Callable[[Array], Array], controller: Callable, X0_lo: Array, X0_hi: Array, n_total_steps: int, reference_seq: Optional[Array] = None) -> Tuple[Array, Array, Array, Array]:
        """Verify with learned dynamics model (overrides self.rhs)."""
        self.rhs = rhs
        self.dyn_reach._set_rhs(rhs)
        self._set_crown_nn(controller)
        return self.verify(X0_lo, X0_hi, n_total_steps, reference_seq)

class OpenLoopSystem_dyn(irx.OpenLoopSystem):
    def __init__(self, state_dim, rhs) -> None:
        self.evolution = "continuous"
        self.xlen = state_dim
        self.rhs = rhs

    def f(
        self, t: jnp.ndarray, x: jnp.ndarray
    ) -> jnp.ndarray:
        return self.rhs(x)[:self.xlen]

class OpenLoopSystem(irx.OpenLoopSystem):
    def __init__(self, state_dim, rhs) -> None:
        self.evolution = "continuous"
        self.xlen = state_dim
        self.rhs = rhs

    def f(
        self, t: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray, w: jnp.ndarray
    ) -> jnp.ndarray:
        z = jnp.concatenate([x, u], axis=-1)
        return self.rhs(z)[:self.xlen]
        # px, py, psi, v = x.ravel()
        # u1 = u.ravel()
        # beta = jnp.arctan(jnp.tan(u1) / 2)
        # return jnp.array(
        #     [v * jnp.cos(psi + beta), v * jnp.sin(psi + beta), v * jnp.sin(beta), u1]
        # )

        # x1,x2,x3,x4,u1 = z
        # dx1 = x2
        # dx2 = 2 * u1
        # dx3 = x4
        # dx4 = (0.08*0.41*(9.8 * jnp.sin(x3) - 2*u1 * jnp.cos(x3)) - 0.0021 * x4) / 0.0105
        # return jnp.stack([dx1, dx2, dx3, dx4], axis=0)
