
from __future__ import annotations
from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp

from src.taylor_model import Interval, QuadPoly, QuadTM
import src.settings as settings


def remainder_picard(rhs_tm_fn: Callable,
                     new_x0: QuadTM,
                        x_tm_seed: QuadTM,
                        h: float,
                        step_lo,
                        step_hi,
                        *,
                        rounds: int,
                        stop_ratio: float = 0.95) -> QuadTM:
    # x_tm_cur = x_tm_seed.truncate_to_affine(step_lo, step_hi)
    x_tm_cur = x_tm_seed
    rhs_tm: QuadTM = rhs_tm_fn(x_tm_cur, step_lo, step_hi)
    delta     = rhs_tm.integrate_time(h, step_lo, step_hi)
    x_tm_next = new_x0.add(delta)
    init_shrinked = x_tm_next.R.subseteq_elem(x_tm_cur.R)
    if settings.CONFIG["DEBUG_LOG"]:
        jax.debug.print(f"Initial Picard shrinked: {init_shrinked.all()}")

    roundoff_r = x_tm_next.P.sub(x_tm_cur.P).eval_interval(step_lo, step_hi)
    # x_tm_cur.R = x_tm_next.R.add(roundoff_r)
    
    def body(x_tm_cur: QuadTM, _):
        # if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
        #     x_tm_cur = x_tm_cur.truncate_to_affine(step_lo, step_hi)
        rhs_tm: QuadTM = rhs_tm_fn(x_tm_cur, step_lo, step_hi)
        # rhs_tm = rhs_tm.truncate_to_affine(step_lo, step_hi)
        # x_tm_cur.log(prefix="x_tm_cur", dim=None)
        # rhs_tm.log(prefix="rhs_tm", dim=None)
        delta     = rhs_tm.integrate_time(h, step_lo, step_hi)
        x_tm_next = new_x0.add(delta)
        if settings.CONFIG["DEBUG_LOG"]:
            x_tm_cur.log("Remainder Picard current", dim=settings.CONFIG["CHECK_DIM"])
            x_tm_next.R.log("Remainder Picard step", dim=settings.CONFIG["CHECK_DIM"])

        next_R = x_tm_next.R.add(roundoff_r)
        shrinked = next_R.subseteq_elem(x_tm_cur.R)
        x_tm_next.R.lo = jnp.where(shrinked, next_R.lo, x_tm_cur.R.lo)
        x_tm_next.R.hi = jnp.where(shrinked, next_R.hi, x_tm_cur.R.hi)

        return x_tm_next, None
        x_tm_cur.R = x_tm_next.R

        return x_tm_cur, None

    x_tm_final, _ = jax.lax.scan(body, x_tm_cur, xs=None, length=rounds, unroll=False)
    return x_tm_final, init_shrinked

    #     return x_tm_cur, shrinked

    # x_tm_final, shrinked = jax.lax.scan(body, x_tm_cur, xs=None, length=rounds, unroll=False)
    # all_shrinked = init_shrinked & jnp.all(shrinked, axis=0)
    # return x_tm_final, all_shrinked
