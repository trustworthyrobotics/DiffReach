
from __future__ import annotations
from typing import Callable, Tuple, Optional

import jax
import jax.numpy as jnp

import src.settings as settings
from src.taylor_model import Interval, QuadPoly, QuadTM
from src.crown_wrapper import crown
from src.rhs_eval import build_auto_rhs_analytic
from src.rhs_eval_nn import build_auto_rhs_nn
from src.picard import remainder_picard
from src.symbolic_remainder import (
    SymbolicRemainderState,
    init_symbolic_state,
    symbolic_step_linear,
)

from jax_verify import IntervalBound, backward_crown_bound_propagation

from src.systems import CT_Dyn_Sys, CT_Ctl_Sys, CT_Plan_Sys, _make_tgrid, DT_Dyn_Sys, DT_Ctl_Sys, DT_Plan_Sys

Array = jax.Array

def init_remainder_abs(x_poly_only: QuadTM, eps: float) -> QuadTM:
    """
    Absolute seeding: add a tiny symmetric interval ±eps to each state dim.
    Args:
      x_poly_only: QuadTM whose P carries the polynomial part (R is typically zero before seeding).
      eps: scalar or array broadcastable to (B, D). Units are state units (absolute).
    Returns:
      QuadTM with same P and R expanded by [-eps, +eps].
    """
    B, D = x_poly_only.P.c.shape
    eps = jnp.asarray(eps, dtype=x_poly_only.P.c.dtype)
    eps = jnp.broadcast_to(eps, (B, D))
    new_R = Interval(
        lo=x_poly_only.R.lo - eps,
        hi=x_poly_only.R.hi + eps,
    )
    return QuadTM(x_poly_only.P, new_R)

def build_linear_tm(c: Array, S: Array, dtype=jnp.float32) -> QuadTM:
    """
    Build a degree-1 polynomial P(z)=c + sum_i S_i * y_i (time is index 0, y axes start at 1).
    c,S: [B,D].  Returns QuadTM with Lt=0.
    """
    B, D = c.shape
    V = D + 1
    P = QuadPoly.zeros(B, D, V, dtype=dtype)
    P.c = c.astype(dtype)
    idx = jnp.arange(D)
    # place scales on spatial axes (1..D)
    P.L = P.L.at[:, idx, idx + 1].set(S.astype(dtype))
    return QuadTM.from_poly(P)

def _make_step_boxes(B: int, D: int, h: float, dtype=jnp.float32) -> Tuple[Array, Array, Array, Array]:
    """
    Build step-local boxes:
      step_lo/hi:  t ∈ [0,h], y ∈ [-1,1]
      eval_lo/hi:  t fixed at 0, y ∈ [-1,1]
    """
    zeros = jnp.zeros((B, 1), dtype=dtype)
    hcol  = jnp.full((B, 1), h, dtype=dtype)
    ones  = jnp.ones((B, D), dtype=dtype)
    step_lo = jnp.concatenate([zeros, -ones], axis=1)
    step_hi = jnp.concatenate([hcol,  +ones], axis=1)
    eval_lo = jnp.concatenate([zeros, -ones], axis=1)
    eval_hi = jnp.concatenate([zeros,  +ones], axis=1)
    return step_lo, step_hi, eval_lo, eval_hi

def identity_parameterization(B: int, D: int, V: int, dtype=jnp.float32) -> QuadTM:
    """
    Parameterization result TMV: x = I * y  (time index 0 unused), constants zero, Lt zero.
    """
    P = QuadPoly.zeros(B, D, V, dtype=dtype)
    idx = jnp.arange(D)
    P.L = P.L.at[:, idx, idx + 1].set(1.0)
    return QuadTM.from_poly(P)


class CT_Dyn_Reach(CT_Dyn_Sys):
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
        self.init_remainder = jnp.broadcast_to(jnp.asarray(init_remainder), (state_dim,))
        self.frr_rounds = frr_rounds
        self.frr_stop_ratio = frr_stop_ratio
        self.sr_window_size = sr_window_size

        self.D = self.state_dim
        self.V = self.D + 1
        # Build RHS interpreters and remainder-Picard operator once.
        self._set_rhs(rhs)

    def _set_rhs(self, rhs: Callable[[Array], Array]):
        """Update RHS and rebuild interpreters."""
        if rhs is None:
            self.rhs = self.rhs_poly_fn = self.rhs_tm_fn = None
            return
        self.rhs = rhs
        if self.nn_dyn:
            self.rhs_poly_fn, self.rhs_tm_fn = build_auto_rhs_nn(self.rhs, D=self.D, V=self.V)
        else:
            self.rhs_poly_fn, self.rhs_tm_fn = build_auto_rhs_analytic(self.rhs, D=self.D, V=self.V)

    def step_once(self, carry, _) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
        """Advance one ODE step and return ((x_tm_new, result_norm_new),(lo,hi))."""

        x_tm, tmv, sr_state = carry
        x_tm: QuadTM
        tmv: QuadTM
        sr_state: SymbolicRemainderState

        # 0) Step-local boxes and parameterization
        step_lo, step_hi, eval_lo, eval_hi = self.step_boxes
        # 1) substitute previous finals at t:=h and get constants
        x0_tm = x_tm.evaluate_time(self.step_size) # x(t+h) w.r.t current seed
        c = x0_tm.P.c  # [B,D]
        # 2) compose P(h,x) with parameterization and normalize (get scales S)
        if settings.CONFIG["DEBUG_LOG"]:
            jax.debug.print("=== before ===")
            x0_tm.log(prefix="tmv_of_x0", dim=settings.CONFIG["CHECK_DIM"])
            tmv.log(prefix="tmv", dim=settings.CONFIG["CHECK_DIM"])

        S, result_norm, sr_next = symbolic_step_linear(tmv, x0_tm, sr_state, eval_lo, eval_hi)          # returns (S, normalized tmv)
        if settings.CONFIG["DEBUG_LOG"]:
            jax.debug.print(f"c = {c}")
            jax.debug.print(f"S = {S}")
            result_norm.log(prefix="result.tmv (final)", dim=settings.CONFIG["CHECK_DIM"])
            jax.debug.print("=== after ===")
        # 3) affine seed for next finals: x = c + S*y
        new_x0 = build_linear_tm(c, S)

        # 4-5) Two Picard trunc-2 (poly-only)
        baseP = new_x0.P
        poly1 = self.rhs_poly_fn(baseP, step_lo, step_hi).integrate_time_trunc()
        poly1   = baseP.add(poly1)
        # if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
        #     poly1 = poly1.truncate_to_affine(step_lo, step_hi)

        poly2 = self.rhs_poly_fn(poly1, step_lo, step_hi).integrate_time_trunc()
        poly2  = baseP.add(poly2)
        # if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
        #     poly2 = poly2.truncate_to_affine(step_lo, step_hi)
        x_tm_poly = QuadTM.from_poly(poly2)

        # 6) seed remainder (absolute)
        # prev_remainder = jnp.maximum(jnp.abs(x_tm.R.hi), jnp.abs(x_tm.R.lo))
        # init_remainder = jnp.minimum(self.init_remainder, prev_remainder)
        init_remainder = self.init_remainder
        x_tm_seed = init_remainder_abs(x_tm_poly, init_remainder)
        if settings.CONFIG["DEBUG_LOG"]:
            new_x0.log(prefix="new_x0", dim=settings.CONFIG["CHECK_DIM"])
            # x_tm_seed.P.log(prefix="initial poly", dim=settings.CONFIG["CHECK_DIM"])
            # x_tm_seed.R.log(prefix="initial remainder", dim=settings.CONFIG["CHECK_DIM"])
            # jax.debug.print(f"step_lo = {step_lo}")
            # jax.debug.print(f"step_hi = {step_hi}")

        # 7) refine remainder (Picard)
        x_tm, init_shrinked = remainder_picard(self.rhs_tm_fn, new_x0, x_tm_seed, self.step_size, step_lo, step_hi, rounds=self.frr_rounds, stop_ratio=self.frr_stop_ratio)
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            x_tm = x_tm.truncate_to_affine(step_lo, step_hi)
        # 8) interval over this step
        # iv = x_tm.eval_interval(step_lo, step_hi)
        if settings.CONFIG["BOUND_TIME_STEP"]:
            # set time to [h,h] for step bounding
            step_lo_ = jnp.concatenate([step_hi[:, :1], step_lo[:, 1:]], axis=1)
            iv = x_tm.compose_affine(result_norm, self.step_size).eval_interval(step_lo_, step_hi)
        else:
            iv = x_tm.compose_affine(result_norm, self.step_size).eval_interval(step_lo, step_hi)
        if settings.CONFIG["DEBUG_LOG"]:
            # new_x0.log(prefix="new_x0")
            x_tm.log(prefix="final x", dim=settings.CONFIG["CHECK_DIM"])
        return (x_tm, result_norm, sr_next), (iv.lo, iv.hi, init_shrinked)

    # ---- public entrypoints ----
    def verify(self, X0_lo: Array, X0_hi: Array, n_total_steps: int) -> Tuple[Array, Array, Array, Array]:
        """Over-approximate reachable sets for a continuous-time autonomous system.

        Args:
          X0_lo, X0_hi: Arrays of shape `(B, state_dim)` describing `B` initial boxes.
          n_total_steps: Number of integration steps.

        Returns:
          A tuple `(times, lowers, uppers, xF, init_shrinked)` where:
          `times` has shape `(n_total_steps + 1,)`,
          `lowers` and `uppers` have shape `(B, n_total_steps + 1, state_dim)`,
          `xF` is the final Taylor model state, and
          `init_shrinked` records Picard shrinkage information per step.
        """
        # Normalize, build boxes & initial models
        B = X0_lo.shape[0]
        self.step_boxes = _make_step_boxes(B=B, D=self.D, h=self.step_size)
        result_0 = identity_parameterization(B, self.D, self.V)  # x = I*y (normalized)
        c_0 = 0.5 * (X0_lo + X0_hi)
        S_0 = 0.5 * (X0_hi - X0_lo)
        x_tm0 = build_linear_tm(c_0, S_0)

        sr0 = init_symbolic_state(B, self.D, M=min(self.sr_window_size, n_total_steps))

        # Initial interval (matches reference driver semantics: evaluate at step box 0)
        step_lo, step_hi, *_ = self.step_boxes
        iv0 = x_tm0.eval_interval(step_lo, step_hi)

        # Scan the cell
        times  = _make_tgrid(n_total_steps, self.step_size)
        (xF, _, _), (los, his, init_shrinked) = jax.lax.scan(self.step_once, (x_tm0, result_0, sr0), None, length=n_total_steps)
        lowers = jnp.concatenate([iv0.lo[:, None, ...], los.transpose((1, 0, 2))], axis=1)
        uppers = jnp.concatenate([iv0.hi[:, None, ...], his.transpose((1, 0, 2))], axis=1)
        # lowers, uppers: (B, n_total_steps+1, D)
        return times, lowers, uppers, xF, init_shrinked

def impose_control_w_crown(new_x0: QuadTM,
                                     crown_nn: Callable,
                                     eval_lo: jax.Array,
                                     eval_hi: jax.Array,
                                     D_s: int,
                                     D_a: int, reference: Optional[Array] = None
                                     ) -> QuadTM:
    dtype = eval_lo.dtype
    R_lo = new_x0.R.lo[:, :D_s]
    R_hi = new_x0.R.hi[:, :D_s]
    if reference is None:
        input_lo = jnp.concatenate([eval_lo, R_lo], axis=1) # [B,V+D]
        input_hi = jnp.concatenate([eval_hi, R_hi], axis=1)
    else:
        input_lo = jnp.concatenate([eval_lo, jnp.zeros_like(reference, dtype=dtype), R_lo], axis=1) # [B,V+D_ref+D_s]
        input_hi = jnp.concatenate([eval_hi, jnp.zeros_like(reference, dtype=dtype), R_hi], axis=1)
    if settings.CONFIG["FP64_IN_CROWN"]:
        input_lo = input_lo.astype(jnp.float64)
        input_hi = input_hi.astype(jnp.float64)

    affine_weight = new_x0.P.L[:, :D_s]   # [B,D_s,V (D_s+D_a+1)]
    affine_coeff = new_x0.P.c[:, :D_s] # [B,D_s]

    if reference is not None:
        affine_weight = jnp.concatenate([affine_weight, jnp.zeros((affine_weight.shape[0], reference.shape[1], affine_weight.shape[2]), dtype=dtype)], axis=1) # [B,D_s+D_ref,V]
        affine_coeff = jnp.concatenate([affine_coeff, reference], axis=1) # [B,D_s+D_ref]

    if settings.CONFIG["FP64_IN_CROWN"]:
        affine_weight = affine_weight.astype(jnp.float64)
        affine_coeff = affine_coeff.astype(jnp.float64)

    def _one(lo, hi, a_weight, a_coeff):
        res = crown_nn(Interval(lo, hi), a_weight, a_coeff)   # -> CROWNResult(lC,uC,ld,ud)
        # Use lC as the shared slope; ld/ud as lower/upper biases.
        T  = res.lC   # [D_a, V]
        bl = res.ld                  # [D_a]
        bu = res.ud                  # [D_a]
        return T, bl, bu
    T, bl, bu = jax.vmap(_one)(input_lo, input_hi, affine_weight, affine_coeff)             # T:[B,D_a,V+D_ref+D_s]

    T=T.astype(dtype)
    bl=bl.astype(dtype)
    bu=bu.astype(dtype)

    T_r = T[:, :, -D_s:]
    T_r_pos = jnp.where(T_r>=0, T_r, jnp.zeros_like(T_r))
    T_r_neg = jnp.where(T_r<0, T_r, jnp.zeros_like(T_r))
    bl = bl + (T_r_pos @ R_lo[:, :, None] + T_r_neg @ R_hi[:, :, None]).squeeze(-1)  # [B,D]
    bu = bu + (T_r_pos @ R_hi[:, :, None] + T_r_neg @ R_lo[:, :, None]).squeeze(-1)  # [B,D]

    # debug: zero action
    # B,D_s = X_box.lo.shape[0], X_box.lo.shape[1]
    # T = jnp.zeros((B, D_a, D_s), dtype=jnp.float64)
    # bl = jnp.zeros((B, D_a), dtype=jnp.float64)
    # bu = jnp.zeros((B, D_a), dtype=jnp.float64)

    # Bias center and slack
    c_vec = 0.5 * (bl + bu)                          # [B,D_a]
    r_vec = 0.5 * (bu - bl)                          # [B,D_a]

    # Remainder propagation + slack

    # Overwrite last D_a coords
    P = new_x0.P
    P.c = P.c.at[:, D_s:].set(c_vec)
    P.L = P.L.at[:, D_s:].set(T[:, :, :P.L.shape[2]])

    R = new_x0.R
    R.lo = R.lo.at[:, D_s:].set(-r_vec)
    R.hi = R.hi.at[:, D_s:].set(r_vec)

    return QuadTM(P, R)

class CT_Ctl_Reach(CT_Ctl_Sys):
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
        # Inner ODE reach core over augmented state
        self.dyn_reach = CT_Dyn_Reach(
            rhs=self.rhs,                # augmented-state RHS with du=0
            state_dim=self.D,
            nn_dyn=self.nn_dyn,
            step_size=self.step_size,
            init_remainder=init_remainder,
            frr_rounds=frr_rounds,
            frr_stop_ratio=frr_stop_ratio,
            sr_window_size=sr_window_size,
        )
        self._set_crown_nn(controller)

        # Actions in control system is constant during control interval, so no remainder in action dims.
        self.dyn_reach.init_remainder = self.dyn_reach.init_remainder.at[-self.Da:].set(0.0)

    def _set_crown_nn(self, controller: Callable):
        crown_nn = crown(controller, in_len=self.V, out_len=self.Da, enable_r=True)
        self.crown_nn = crown_nn
        return

    def _normalize_reference_seq_for_verify(
        self,
        reference_seq: Optional[Array],
        *,
        batch_size: int,
        n_control_steps: int,
        dtype,
    ) -> Array:
        if reference_seq is None:
            return jnp.zeros((batch_size, n_control_steps, self.Dr), dtype=dtype)

        reference_seq = jnp.asarray(reference_seq, dtype=dtype)
        shared_shape = (n_control_steps, self.Dr)
        batched_shape = (batch_size, n_control_steps, self.Dr)
        if reference_seq.shape == shared_shape:
            return jnp.broadcast_to(reference_seq[None, :, :], batched_shape)
        if reference_seq.shape != batched_shape:
            raise ValueError(
                "reference_seq must have shape "
                f"{shared_shape} or {batched_shape}, got {reference_seq.shape}."
            )
        return reference_seq

    def verify(self, Z0_lo: Array, Z0_hi: Array,
     n_total_steps: int, reference_seq: Optional[Array] = None) -> Tuple[Array, Array, Array, Array]:
        """Over-approximate reachable sets for a continuous-time controlled system.

        Args:
          Z0_lo, Z0_hi: Arrays of shape `(B, state_dim + action_dim)` describing
            `B` initial augmented-state boxes.
          n_total_steps: Number of integration steps.
          reference_seq: Optional reference sequence. Accepted shapes are
            `(n_control_steps, reference_dim)` for a reference shared by all
            partitions or `(B, n_control_steps, reference_dim)` for per-partition
            references. If omitted, a zero reference is used.

        Returns:
          A tuple `(times, lowers, uppers, xF, init_shrinked)` where:
          `times` has shape `(n_total_steps + 1,)`,
          `lowers` and `uppers` have shape `(B, n_total_steps + 1, state_dim + action_dim)`,
          `xF` is the final Taylor model state, and
          `init_shrinked` records Picard shrinkage information per step.
        """
        B = Z0_lo.shape[0]
        D = self.D
        V = self.V
        n_control_steps = round(n_total_steps / self.n_steps_per_control)
        reference_seq = self._normalize_reference_seq_for_verify(
            reference_seq,
            batch_size=B,
            n_control_steps=n_control_steps,
            dtype=Z0_lo.dtype,
        )
        # Prepare inner step boxes and initial parameterization
        self.dyn_reach.step_boxes = _make_step_boxes(B=B, D=D, h=self.step_size)
        result_0 = identity_parameterization(B, D, V)
        c0 = 0.5 * (Z0_lo + Z0_hi)
        s0 = 0.5 * (Z0_hi - Z0_lo)
        x_tm0 = build_linear_tm(c0, s0)

        # Initial interval at step box
        step_lo, step_hi, eval_lo, eval_hi = self.dyn_reach.step_boxes
        iv0 = x_tm0.eval_interval(step_lo, step_hi)

        sr0 = init_symbolic_state(B, D, M=min(self.dyn_reach.sr_window_size, n_total_steps))

        def control_cell(carry, xs):
            x_tm_pre, result_tmv, sr_state = carry
            reference = xs
            x_tm_pre: QuadTM
            result_tmv: QuadTM
            sr_state: SymbolicRemainderState
            # 1) Boundary state box at t=0 (state dims only)
            x0_tm = x_tm_pre.evaluate_time(self.step_size) 
            X_box = x0_tm.eval_interval(eval_lo, eval_hi)
            X_box = Interval(X_box.lo[:, :self.Ds], X_box.hi[:, :self.Ds])  # keep states only
            if settings.CONFIG["DEBUG_LOG"]:
                jax.debug.print("=== control step ===")
                x0_tm.log(prefix="x0_tm (before control)", dim=settings.CONFIG["CHECK_DIM"])
                X_box.log(prefix="X_box (at control boundary)", dim=settings.CONFIG["CHECK_DIM"])
            # 2) Impose controller on the step seed with affine u = T x + c ± r from CROWN
            # x0_tm = impose_control_w_crown(self.crown_nn, X_box, x0_tm, self.Ds, self.Da, reference)
            x0_tm = impose_control_w_crown(x0_tm, self.crown_nn, eval_lo, eval_hi, self.Ds, self.Da, reference)
            if settings.CONFIG["DEBUG_LOG"]:
                x0_tm.log(prefix="x0_tm (after control)", dim=settings.CONFIG["CHECK_DIM"])
                u_box = x0_tm.eval_interval(step_lo, step_hi)
                jax.debug.print(f"u_box = {u_box}")
            # 3) Run K inner ODE steps via CT_Dyn_Reach.step_once
            (x_after, result_after, sr_next), (los, his, init_shrinked) = jax.lax.scan(self.dyn_reach.step_once, (x0_tm, result_tmv, sr_state), None, length=self.n_steps_per_control)
            return (x_after, result_after, sr_next), (los, his, init_shrinked)

        # Outer control scan
        (xF, _, _), (los_all, his_all, init_shrinked_all) = jax.lax.scan(control_cell, (x_tm0, result_0, sr0), reference_seq.transpose((1, 0, 2)), length=n_control_steps)
        # Stitch intervals
        lowers = jnp.concatenate([iv0.lo[:, None, ...], los_all.reshape(-1, B, D).transpose((1, 0, 2))], axis=1)
        uppers = jnp.concatenate([iv0.hi[:, None, ...], his_all.reshape(-1, B, D).transpose((1, 0, 2))], axis=1)
        # lowers, uppers: (B, n_total_steps+1, D)
        times  = _make_tgrid(n_total_steps, self.step_size)
        return times, lowers, uppers, xF, init_shrinked_all

    def verify_w_model(self, rhs: Callable[[Array], Array], controller: Callable, X0_lo: Array, X0_hi: Array, n_total_steps: int, reference_seq: Optional[Array] = None) -> Tuple[Array, Array, Array, Array]:
        """Verify with learned dynamics model (overrides self.rhs)."""
        self.rhs = rhs
        self.dyn_reach._set_rhs(rhs)
        self._set_crown_nn(controller)
        return self.verify(X0_lo, X0_hi, n_total_steps, reference_seq)


def impose_plan(x0_tm: QuadTM, action: Array, D_a: int) -> QuadTM:
    """Impose open-loop action plan on the last D_a coords of the step seed (affine TM)."""
    # action: (B, D_a)
    x0_tm.P.c = x0_tm.P.c.at[:, -D_a:].set(action)
    x0_tm.P.L = x0_tm.P.L.at[:, -D_a:, :].set(0.0)
    x0_tm.P.Lt = x0_tm.P.Lt.at[:, -D_a:, :].set(0.0)
    x0_tm.R.lo = x0_tm.R.lo.at[:, -D_a:].set(0.0)
    x0_tm.R.hi = x0_tm.R.hi.at[:, -D_a:].set(0.0)
    return x0_tm

class CT_Plan_Reach(CT_Plan_Sys):
    """Open-loop planned CT system reachability.

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
                 init_remainder: float = 1e-1,
                 frr_rounds: int = 3,
                 frr_stop_ratio: float = 0.95,
                 sr_window_size: int = 100,
                 config: dict = None,
                 *args, **kwargs
                 ):
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
        self.dyn_reach = CT_Dyn_Reach(
            rhs=self.rhs,                # augmented-state RHS with du=0
            state_dim=self.D,
            nn_dyn=self.nn_dyn,
            step_size=self.step_size,
            init_remainder=init_remainder,
            frr_rounds=frr_rounds,
            frr_stop_ratio=frr_stop_ratio,
            sr_window_size=sr_window_size,
        )
        # Actions in planning system is given by reference, so no remainder in action dims.
        self.dyn_reach.init_remainder = self.dyn_reach.init_remainder.at[-self.Da:].set(0.0)

    def _normalize_action_seq_for_verify(
        self,
        action_seq: Array,
        *,
        batch_size: int,
        n_plan_steps: int,
        dtype,
    ) -> Array:
        action_seq = jnp.asarray(action_seq, dtype=dtype)
        if action_seq.ndim == 3:
            if action_seq.shape[2] != self.Da or action_seq.shape[1] != n_plan_steps:
                raise ValueError(
                    "action_seq must have shape "
                    f"(n_plan, {n_plan_steps}, {self.Da}) or "
                    f"({batch_size}, n_plan, {n_plan_steps}, {self.Da}), got {action_seq.shape}."
                )
            n_plan = action_seq.shape[0]
            return jnp.broadcast_to(action_seq[None, ...], (batch_size, n_plan, n_plan_steps, self.Da))
        if action_seq.ndim != 4 or action_seq.shape[0] != batch_size or action_seq.shape[2] != n_plan_steps or action_seq.shape[3] != self.Da:
            raise ValueError(
                "action_seq must have shape "
                f"(n_plan, {n_plan_steps}, {self.Da}) or "
                f"({batch_size}, n_plan, {n_plan_steps}, {self.Da}), got {action_seq.shape}."
            )
        return action_seq

    def verify(self, Z0_lo: Array, Z0_hi: Array, n_total_steps: int, action_seq: Array):
        """Over-approximate reachable sets for continuous-time open-loop plans.

        Args:
          Z0_lo, Z0_hi: Arrays of shape `(B, state_dim + action_dim)` describing
            `B` initial augmented-state boxes.
          n_total_steps: Total number of integration steps.
          action_seq: Action plans with shape `(n_plan, n_plan_steps, action_dim)`
            or `(B, n_plan, n_plan_steps, action_dim)`, where
            `n_total_steps == n_plan_steps * n_steps_per_plan`.

        Returns:
          A tuple `(times, lowers, uppers, xF, init_shrinked)` where:
          `times` has shape `(n_total_steps + 1,)`,
          `lowers` and `uppers` have shape
          `(B, n_plan, n_total_steps + 1, state_dim + action_dim)`,
          `xF` is the final Taylor model state, and
          `init_shrinked` records shrinkage information per step.
        """
        n_partition = Z0_lo.shape[0]
        action_seq = self._normalize_action_seq_for_verify(
            action_seq,
            batch_size=n_partition,
            n_plan_steps=n_total_steps // self.n_steps_per_plan,
            dtype=Z0_lo.dtype,
        )

        n_plan = action_seq.shape[1]
        n_partition = Z0_lo.shape[0]
        Z0_lo = Z0_lo.repeat(n_plan, axis=0)    # (n_partition*n_plan, D)
        Z0_hi = Z0_hi.repeat(n_plan, axis=0)    # (n_partition*n_plan, D)
        action_seq = action_seq.reshape(n_plan*n_partition, -1, self.Da).transpose((1, 0, 2))  # (n_horizon, n_partition*n_plan, action_dim)
        
        B = Z0_lo.shape[0]
        D = self.D
        V = self.V

        # Prepare inner step boxes and initial parameterization
        self.dyn_reach.step_boxes = _make_step_boxes(B=B, D=D, h=self.step_size)
        result_0 = identity_parameterization(B, D, V)
        c0 = 0.5 * (Z0_lo + Z0_hi)
        s0 = 0.5 * (Z0_hi - Z0_lo)
        x_tm0 = build_linear_tm(c0, s0)

        # Initial interval at step box
        step_lo, step_hi, eval_lo, eval_hi = self.dyn_reach.step_boxes
        iv0 = x_tm0.eval_interval(step_lo, step_hi)

        sr0 = init_symbolic_state(B, D, M=min(self.dyn_reach.sr_window_size, n_total_steps))

        def plan_cell(carry, action):
            x_tm_pre, result_tmv, sr_state = carry
            x_tm_pre: QuadTM
            result_tmv: QuadTM
            sr_state: SymbolicRemainderState
            # 1) Boundary state box at t=0 (state dims only)
            x0_tm = x_tm_pre.evaluate_time(self.step_size) 
            X_box = x0_tm.eval_interval(eval_lo, eval_hi)
            X_box = Interval(X_box.lo[:, :self.Ds], X_box.hi[:, :self.Ds])  # keep states only
            if settings.CONFIG["DEBUG_LOG"]:
                jax.debug.print("=== plan step ===")
                x0_tm.log(prefix="x0_tm (before plan)", dim=settings.CONFIG["CHECK_DIM"])
                X_box.log(prefix="X_box (at plan boundary)", dim=settings.CONFIG["CHECK_DIM"])
            # 2) Impose controller on the step seed with affine u = T x + c ± r from CROWN
            x0_tm = impose_plan(x0_tm, action, self.Da)
            if settings.CONFIG["DEBUG_LOG"]:
                x0_tm.log(prefix="x0_tm (after plan)", dim=settings.CONFIG["CHECK_DIM"])
                u_box = x0_tm.eval_interval(step_lo, step_hi)
                jax.debug.print(f"u_box = {u_box}")
            # 3) Run K inner ODE steps via CT_Dyn_Reach.step_once
            # los, his: (n_steps_per_plan, B, D)
            (x_after, result_after, sr_next), (los, his, init_shrinked) = jax.lax.scan(self.dyn_reach.step_once, (x0_tm, result_tmv, sr_state), None, length=self.n_steps_per_plan)
            return (x_after, result_after, sr_next), (los, his, init_shrinked)

        # Outer plan scan
        # los_all, his_all: (n_horizon, n_steps_per_plan, n_partition*n_plan,  D)
        (xF, _, _), (los_all, his_all, init_shrinked_all) = jax.lax.scan(plan_cell, (x_tm0, result_0, sr0), action_seq)
        # Stitch intervals
        lowers = jnp.concatenate([iv0.lo[:, None, ...], los_all.reshape(-1, B, D).transpose((1, 0, 2))], axis=1)
        uppers = jnp.concatenate([iv0.hi[:, None, ...], his_all.reshape(-1, B, D).transpose((1, 0, 2))], axis=1)
        
        lowers = lowers.reshape(n_partition, n_plan, -1, D)
        uppers = uppers.reshape(n_partition, n_plan, -1, D)
        times  = _make_tgrid(n_total_steps, self.step_size)
        return times, lowers, uppers, xF, init_shrinked_all

    def verify_w_model(self, rhs: Callable[[Array], Array], X0_lo: Array, X0_hi: Array, n_total_steps: int, action_seq: Array) -> Tuple[Array, Array, Array, Array]:
        """Verify with learned dynamics model (overrides self.rhs)."""
        self.rhs = rhs
        self.dyn_reach._set_rhs(rhs)
        return self.verify(X0_lo, X0_hi, n_total_steps, action_seq)

class DT_Dyn_Reach(DT_Dyn_Sys):
    """Flow*-style reachability for discrete-time systems x_{k+1} = f(x_k).

    Key difference vs CT_Dyn_Reach:
      - No time variable, no integrate_time_trunc(), no Picard integral.
      - Each step computes an enclosure of f applied to the current Taylor model:
            x_tm_next ≈ rhs( x_tm_current )
        with remainder refinement if desired.
    """

    def __init__(self,
                 rhs: Callable[[Array], Array],  # or Callable[[Array],Array] if no control
                 state_dim: int,
                 nn_dyn: bool,
                 step_size: int,
                 sr_window_size: int = 5,
                 config: dict = None,
                 *args, **kwargs
                 ):
        super().__init__(rhs=rhs, state_dim=state_dim, step_size=step_size)
        assert step_size == 1, "For DT_Dyn_Reach, step_size must be 1."
        self.nn_dyn = nn_dyn
        if config is not None:
            settings.update_config(config)
        self.sr_window_size = sr_window_size

        self.D = self.state_dim
        self.V = self.D + 1  # keep your convention; can also be D if you prefer

        # Build wrappers that accept Poly/TM inputs and return Poly/TM outputs for the map f.
        self._set_rhs(rhs)

    def _set_rhs(self, rhs: Callable[[Array], Array]):
        """Update RHS and rebuild interpreters."""
        if rhs is None:
            self.rhs = self.rhs_poly_fn = self.rhs_tm_fn = None
            return
        self.rhs = rhs
        if self.nn_dyn:
            self.rhs_poly_fn, self.rhs_tm_fn = build_auto_rhs_nn(self.rhs, D=self.D, V=self.V)
        else:
            self.rhs_poly_fn, self.rhs_tm_fn = build_auto_rhs_analytic(self.rhs, D=self.D, V=self.V)

    def step_once(self, carry, _) -> ...:
        x_tm, tmv, sr_state = carry 
        step_lo, step_hi, eval_lo, eval_hi = self.step_boxes
        # 1) Evaluate Dynamics directly on the previous Polynomial
        # This preserves the 'curve' of the set.
        # Note: We skip 'symbolic_step_linear' / 'build_linear_tm'
        x_next = self.rhs_tm_fn(x_tm, eval_lo, eval_hi)
        # 3) Logging (Evaluate bounds)
        iv = x_next.eval_interval(eval_lo, eval_hi)
        
        return (x_next, tmv, sr_state), (iv.lo, iv.hi, jnp.ones_like(x_tm.P.c))  # placeholder for init_shrinked

    # -----------------------------
    # Public entrypoint
    # -----------------------------
    def verify(self, X0_lo: Array, X0_hi: Array, n_total_steps: int) -> Tuple[Array, Array, Array, QuadTM, Array]:
        """Over-approximate reachable sets for a discrete-time autonomous system.

        Args:
          X0_lo, X0_hi: Arrays of shape `(B, state_dim)` describing `B` initial boxes.
          n_total_steps: Number of discrete-time steps.

        Returns:
          A tuple `(steps, lowers, uppers, xF, init_shrinked)` where:
          `steps` has shape `(n_total_steps + 1,)`,
          `lowers` and `uppers` have shape `(B, n_total_steps + 1, state_dim)`,
          `xF` is the final Taylor model state, and
          `init_shrinked` stores per-step refinement diagnostics.
        """
        B = X0_lo.shape[0]

        # Step boxes: now only parameter-domain boxes (no time dimension).
        # Convention: y ∈ [-1,1]^V and possibly separate eval box.
        self.step_boxes = _make_step_boxes(B=B, D=self.D, h=self.step_size)
        step_lo, step_hi, *_ = self.step_boxes

        # Initial parameterization and seed TM from the initial box.
        result_0 = identity_parameterization(B, self.D, self.V)  # x = I*y
        c_0 = 0.5 * (X0_lo + X0_hi)
        S_0 = 0.5 * (X0_hi - X0_lo)
        x_tm0 = build_linear_tm(c_0, S_0)

        sr0 = init_symbolic_state(B, self.D, M=min(self.sr_window_size, n_total_steps))

        # Initial interval
        iv0 = x_tm0.eval_interval(step_lo, step_hi)

        # Discrete grid (k = 0..N). If you want "time", you can output k or k*dt elsewhere.
        steps = jnp.arange(n_total_steps + 1)

        (xF, _, _), (los, his, init_shrinked) = jax.lax.scan(
            self.step_once, (x_tm0, result_0, sr0), None, length=n_total_steps
        )

        lowers = jnp.concatenate([iv0.lo[:, None, ...], los.transpose((1, 0, 2))], axis=1)
        uppers = jnp.concatenate([iv0.hi[:, None, ...], his.transpose((1, 0, 2))], axis=1)
        # lowers, uppers: (B, n_total_steps+1, D)
        return steps, lowers, uppers, xF, init_shrinked

class DT_Ctl_Reach(DT_Ctl_Sys):
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
                 step_size: int,
                 reference_dim: int = 0,
                 sr_window_size: int = 100,
                 config: dict = None,
                 *args, **kwargs
                 ):
        assert step_size == 1, "For DT_Ctl_Reach, step_size must be 1."
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
        # Inner ODE reach core over augmented state
        self.dyn_reach = DT_Dyn_Reach(
            rhs=self.rhs,                # augmented-state RHS with du=0
            state_dim=self.D,
            nn_dyn=self.nn_dyn,
            step_size=self.step_size,
            sr_window_size=sr_window_size,
        )
        self._set_crown_nn(controller)

    def _set_crown_nn(self, controller: Callable):
        crown_nn = crown(controller, in_len=self.V, out_len=self.Da, enable_r=True)
        self.crown_nn = crown_nn
        return

    def _normalize_reference_seq_for_verify(
        self,
        reference_seq: Optional[Array],
        *,
        batch_size: int,
        n_control_steps: int,
        dtype,
    ) -> Array:
        if reference_seq is None:
            return jnp.zeros((batch_size, n_control_steps, self.Dr), dtype=dtype)

        reference_seq = jnp.asarray(reference_seq, dtype=dtype)
        shared_shape = (n_control_steps, self.Dr)
        batched_shape = (batch_size, n_control_steps, self.Dr)
        if reference_seq.shape == shared_shape:
            return jnp.broadcast_to(reference_seq[None, :, :], batched_shape)
        if reference_seq.shape != batched_shape:
            raise ValueError(
                "reference_seq must have shape "
                f"{shared_shape} or {batched_shape}, got {reference_seq.shape}."
            )
        return reference_seq

    def verify(self, Z0_lo: Array, Z0_hi: Array, n_total_steps: int, reference_seq: Optional[Array] = None):
        """Over-approximate reachable sets for a discrete-time controlled system.

        Args:
          Z0_lo, Z0_hi: Arrays of shape `(B, state_dim + action_dim)` describing
            `B` initial augmented-state boxes.
          n_total_steps: Number of discrete-time steps.
          reference_seq: Optional reference sequence. Accepted shapes are
            `(n_control_steps, reference_dim)` for a reference shared by all
            partitions or `(B, n_control_steps, reference_dim)` for per-partition
            references. If omitted, a zero reference is used.

        Returns:
          A tuple `(steps, lowers, uppers, xF, init_shrinked)` where:
          `steps` has shape `(n_total_steps + 1,)`,
          `lowers` and `uppers` have shape `(B, n_total_steps + 1, state_dim + action_dim)`,
          `xF` is the final Taylor model state, and
          `init_shrinked` stores per-step refinement diagnostics.
        """
        B = Z0_lo.shape[0]
        D = self.D
        V = self.V
        n_control_steps = round(n_total_steps / self.n_steps_per_control)
        reference_seq = self._normalize_reference_seq_for_verify(
            reference_seq,
            batch_size=B,
            n_control_steps=n_control_steps,
            dtype=Z0_lo.dtype,
        )
        # Prepare inner step boxes and initial parameterization
        self.dyn_reach.step_boxes = _make_step_boxes(B=B, D=D, h=self.step_size)
        result_0 = identity_parameterization(B, D, V)
        c0 = 0.5 * (Z0_lo + Z0_hi)
        s0 = 0.5 * (Z0_hi - Z0_lo)
        x_tm0 = build_linear_tm(c0, s0)

        # Initial interval at step box
        step_lo, step_hi, eval_lo, eval_hi = self.dyn_reach.step_boxes
        iv0 = x_tm0.eval_interval(step_lo, step_hi)

        sr0 = init_symbolic_state(B, D, M=min(self.dyn_reach.sr_window_size, n_total_steps))

        def control_cell(carry, xs):
            x_tm_pre, result_tmv, sr_state = carry
            reference = xs
            x_tm_pre: QuadTM
            result_tmv: QuadTM
            sr_state: SymbolicRemainderState
            # 1) Boundary state box at t=0 (state dims only)
            x0_tm = x_tm_pre.evaluate_time(self.step_size) 
            X_box = x0_tm.eval_interval(eval_lo, eval_hi)
            X_box = Interval(X_box.lo[:, :self.Ds], X_box.hi[:, :self.Ds])  # keep states only
            if settings.CONFIG["DEBUG_LOG"]:
                jax.debug.print("=== control step ===")
                x0_tm.log(prefix="x0_tm (before control)", dim=settings.CONFIG["CHECK_DIM"])
                X_box.log(prefix="X_box (at control boundary)", dim=settings.CONFIG["CHECK_DIM"])
            # 2) Impose controller on the step seed with affine u = T x + c ± r from CROWN
            x0_tm = impose_control_w_crown(x0_tm, self.crown_nn, eval_lo, eval_hi, self.Ds, self.Da, reference)
            if settings.CONFIG["DEBUG_LOG"]:
                x0_tm.log(prefix="x0_tm (after control)", dim=settings.CONFIG["CHECK_DIM"])
                u_box = x0_tm.eval_interval(step_lo, step_hi)
                jax.debug.print(f"u_box = {u_box}")
            # 3) Run K inner ODE steps via CT_Dyn_Reach.step_once
            (x_after, result_after, sr_next), (los, his, init_shrinked) = jax.lax.scan(self.dyn_reach.step_once, (x0_tm, result_tmv, sr_state), None, length=self.n_steps_per_control)
            return (x_after, result_after, sr_next), (los, his, init_shrinked)

        # Outer control scan
        (xF, _, _), (los_all, his_all, init_shrinked_all) = jax.lax.scan(control_cell, (x_tm0, result_0, sr0), reference_seq.transpose((1, 0, 2)), length=n_control_steps)
        # Stitch intervals
        lowers = jnp.concatenate([iv0.lo[:, None, ...], los_all.reshape(B, -1, D)], axis=1)
        uppers = jnp.concatenate([iv0.hi[:, None, ...], his_all.reshape(B, -1, D)], axis=1)
        # lowers, uppers: (B, n_total_steps+1, D)
        times  = _make_tgrid(n_total_steps, self.step_size)
        return times, lowers, uppers, xF, init_shrinked_all

    def verify_w_model(self, rhs: Callable[[Array], Array], controller: Callable, X0_lo: Array, X0_hi: Array, n_total_steps: int, reference_seq: Optional[Array] = None) -> Tuple[Array, Array, Array, Array]:
        """Verify with learned dynamics model (overrides self.rhs)."""
        self.rhs = rhs
        self.dyn_reach._set_rhs(rhs)
        self._set_crown_nn(controller)
        return self.verify(X0_lo, X0_hi, n_total_steps, reference_seq)

class DT_Plan_Reach(DT_Plan_Sys):
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

    def _normalize_action_seq_for_verify(
        self,
        action_seq: Array,
        *,
        batch_size: int,
        n_plan_steps: int,
        dtype,
    ) -> Array:
        action_seq = jnp.asarray(action_seq, dtype=dtype)
        if action_seq.ndim == 3:
            if action_seq.shape[2] != self.Da or action_seq.shape[1] != n_plan_steps:
                raise ValueError(
                    "action_seq must have shape "
                    f"(n_plan, {n_plan_steps}, {self.Da}) or "
                    f"({batch_size}, n_plan, {n_plan_steps}, {self.Da}), got {action_seq.shape}."
                )
            n_plan = action_seq.shape[0]
            return jnp.broadcast_to(action_seq[None, ...], (batch_size, n_plan, n_plan_steps, self.Da))
        if action_seq.ndim != 4 or action_seq.shape[0] != batch_size or action_seq.shape[2] != n_plan_steps or action_seq.shape[3] != self.Da:
            raise ValueError(
                "action_seq must have shape "
                f"(n_plan, {n_plan_steps}, {self.Da}) or "
                f"({batch_size}, n_plan, {n_plan_steps}, {self.Da}), got {action_seq.shape}."
            )
        return action_seq

    def verify(self, Z0_lo: Array, Z0_hi: Array, n_total_steps: int, action_seq: Array):
        """Over-approximate reachable sets for discrete-time open-loop plans.

        Args:
          Z0_lo, Z0_hi: Arrays of shape `(B, state_dim + action_dim)` describing
            `B` initial augmented-state boxes.
          n_total_steps: Total number of discrete-time steps.
          action_seq: Action plans with shape `(n_plan, n_plan_steps, action_dim)`
            or `(B, n_plan, n_plan_steps, action_dim)`, where
            `n_total_steps == n_plan_steps * n_steps_per_plan`.

        Returns:
          A tuple `(steps, lowers, uppers, xF, init_shrinked)` where:
          `steps` has shape `(n_total_steps + 1,)`,
          `lowers` and `uppers` have shape
          `(B, n_plan, n_total_steps + 1, state_dim + action_dim)`,
          `xF` is the final Taylor model state, and
          `init_shrinked` stores per-step refinement diagnostics.
        """
        n_partition = Z0_lo.shape[0]
        action_seq = self._normalize_action_seq_for_verify(
            action_seq,
            batch_size=n_partition,
            n_plan_steps=n_total_steps // self.n_steps_per_plan,
            dtype=Z0_lo.dtype,
        )

        n_plan = action_seq.shape[1]
        n_partition = Z0_lo.shape[0]
        Z0_lo = Z0_lo.repeat(n_plan, axis=0)    # (n_partition*n_plan, D)
        Z0_hi = Z0_hi.repeat(n_plan, axis=0)    # (n_partition*n_plan, D)
        action_seq = action_seq.reshape(n_plan*n_partition, -1, self.Da).transpose((1, 0, 2))  # (n_horizon, n_partition*n_plan, action_dim)
        
        B = Z0_lo.shape[0]
        D = self.D
        V = self.V

        # Prepare inner step boxes and initial parameterization
        self.dyn_reach.step_boxes = _make_step_boxes(B=B, D=D, h=self.step_size)
        result_0 = identity_parameterization(B, D, V)
        c0 = 0.5 * (Z0_lo + Z0_hi)
        s0 = 0.5 * (Z0_hi - Z0_lo)
        x_tm0 = build_linear_tm(c0, s0)

        # Initial interval at step box
        step_lo, step_hi, eval_lo, eval_hi = self.dyn_reach.step_boxes
        iv0 = x_tm0.eval_interval(step_lo, step_hi)

        sr0 = init_symbolic_state(B, D, M=min(self.dyn_reach.sr_window_size, n_total_steps))

        def plan_cell(carry, action):
            x_tm_pre, result_tmv, sr_state = carry
            x_tm_pre: QuadTM
            result_tmv: QuadTM
            sr_state: SymbolicRemainderState
            # 1) Boundary state box at t=0 (state dims only)
            x0_tm = x_tm_pre.evaluate_time(self.step_size) 
            X_box = x0_tm.eval_interval(eval_lo, eval_hi)
            X_box = Interval(X_box.lo[:, :self.Ds], X_box.hi[:, :self.Ds])  # keep states only
            if settings.CONFIG["DEBUG_LOG"]:
                jax.debug.print("=== plan step ===")
                x0_tm.log(prefix="x0_tm (before plan)", dim=settings.CONFIG["CHECK_DIM"])
                X_box.log(prefix="X_box (at plan boundary)", dim=settings.CONFIG["CHECK_DIM"])
            # 2) Impose controller on the step seed with affine u = T x + c ± r from CROWN
            x0_tm = impose_plan(x0_tm, action, self.Da)
            if settings.CONFIG["DEBUG_LOG"]:
                x0_tm.log(prefix="x0_tm (after plan)", dim=settings.CONFIG["CHECK_DIM"])
                u_box = x0_tm.eval_interval(step_lo, step_hi)
                jax.debug.print(f"u_box = {u_box}")
            # 3) Run K inner ODE steps via CT_Dyn_Reach.step_once
            # los, his: (n_steps_per_plan, B, D)
            (x_after, result_after, sr_next), (los, his, init_shrinked) = jax.lax.scan(self.dyn_reach.step_once, (x0_tm, result_tmv, sr_state), None, length=self.n_steps_per_plan)
            return (x_after, result_after, sr_next), (los, his, init_shrinked)

        # Outer plan scan
        # los_all, his_all: (n_horizon, n_steps_per_plan, n_partition*n_plan,  D)
        (xF, _, _), (los_all, his_all, init_shrinked_all) = jax.lax.scan(plan_cell, (x_tm0, result_0, sr0), action_seq)
        # Stitch intervals
        lowers = jnp.concatenate([iv0.lo[:, None, ...], los_all.reshape(-1, B, D).transpose((1, 0, 2))], axis=1)
        uppers = jnp.concatenate([iv0.hi[:, None, ...], his_all.reshape(-1, B, D).transpose((1, 0, 2))], axis=1)
        
        lowers = lowers.reshape(n_partition, n_plan, -1, D)
        uppers = uppers.reshape(n_partition, n_plan, -1, D)
        times  = _make_tgrid(n_total_steps, self.step_size)
        return times, lowers, uppers, xF, init_shrinked_all

    def verify_w_model(self, rhs: Callable[[Array], Array], X0_lo: Array, X0_hi: Array, n_total_steps: int, action_seq: Array) -> Tuple[Array, Array, Array, Array]:
        """Verify with learned dynamics model (overrides self.rhs)."""
        self.rhs = rhs
        self.dyn_reach._set_rhs(rhs)
        return self.verify(X0_lo, X0_hi, n_total_steps, action_seq)
