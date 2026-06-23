from __future__ import annotations
from typing import Callable, Dict, Any, Tuple

import jax
import jax.numpy as jnp

from src.taylor_model import Interval, QuadPoly, QuadTM
from src.crown_wrapper import crown

import src.settings as settings

Array = jnp.ndarray

def build_nn_rhs_poly(f: Callable, D: int, V: int):
    crown_nn = crown(f, in_len=D, out_len=D)
    def eval_poly(x_poly: QuadPoly, box_lo: Array, box_hi: Array) -> QuadPoly:
        x_box = x_poly.eval_interval(box_lo, box_hi)

        dtype = x_box.lo.dtype
        if settings.CONFIG["FP64_IN_CROWN"]:
            x_box = x_box.to_fp64()
        def _one(lo, hi):
            res = crown_nn(Interval(lo, hi))   # -> CROWNResult(lC,uC,ld,ud)
            # Use lC as the shared slope; ld/ud as lower/upper biases.
            T  = res.lC # [D, D]
            bl = res.ld # [D]
            bu = res.ud # [D]
            return T, bl, bu
        T, bl, bu = jax.vmap(_one)(x_box.lo, x_box.hi)             # T:[B,D,D]
        T = T.astype(dtype)
        bl = bl.astype(dtype)
        bu = bu.astype(dtype)

        # Linear map
        U_c  = jnp.einsum("bji,bi->bj",   T, x_poly.c)             # [B,D]
        U_L  = jnp.einsum("bji,biV->bjV", T, x_poly.L)             # [B,D,V]

        # Bias center and slack
        c_vec = 0.5 * (bl + bu)                          # [B,D]
        r_vec = 0.5 * (bu - bl)                          # [B,D]
        U_c   = U_c + c_vec
        return QuadPoly(U_c, U_L, jnp.zeros_like(U_L))
    return eval_poly

def build_nn_rhs_tm(f: Callable, D: int, V: int):
    crown_nn = crown(f, in_len=D, out_len=D)
    def eval_tm(x_tm: QuadTM, box_lo: Array, box_hi: Array) -> QuadTM:
        x_box = x_tm.eval_interval(box_lo, box_hi)

        dtype = x_box.lo.dtype
        if settings.CONFIG["FP64_IN_CROWN"]:
            x_box = x_box.to_fp64()
        def _one(lo, hi):
            res = crown_nn(Interval(lo, hi))   # -> CROWNResult(lC,uC,ld,ud)
            # Use lC as the shared slope; ld/ud as lower/upper biases.
            T  = res.lC # [D, D]
            bl = res.ld # [D]
            bu = res.ud # [D]
            return T, bl, bu
        T, bl, bu = jax.vmap(_one)(x_box.lo, x_box.hi)             # T:[B,D,D]
        T = T.astype(dtype)
        bl = bl.astype(dtype)
        bu = bu.astype(dtype)

        # Linear map
        x_tm = x_tm.truncate_to_affine(box_lo, box_hi)
        U_c  = jnp.einsum("bji,bi->bj",   T, x_tm.P.c)             # [B,D]
        U_L  = jnp.einsum("bji,biV->bjV", T, x_tm.P.L)             # [B,D,V]

        # Bias center and slack
        c_vec = 0.5 * (bl + bu)                          # [B,D]
        r_vec = 0.5 * (bu - bl)                          # [B,D]
        U_c   = U_c + c_vec

        return QuadTM(QuadPoly(U_c, U_L, jnp.zeros_like(U_L)), x_tm.R.affine(T).enlarge(r_vec))

    return eval_tm

def build_nn_rhs_poly_aff(f: Callable, D: int, V: int):
    crown_nn = crown(f, in_len=V, out_len=D)
    def eval_poly(x_poly: QuadPoly, box_lo: Array, box_hi: Array) -> QuadPoly:
        x_poly = x_poly.truncate_to_affine(box_lo, box_hi)
        affine_weight = x_poly.L
        affine_coeff  = x_poly.c
        input_lo = box_lo
        input_hi = box_hi
        if settings.CONFIG["FP64_IN_CROWN"]:
            affine_weight = affine_weight.astype(jnp.float64)
            affine_coeff  = affine_coeff.astype(jnp.float64)
            input_lo = input_lo.astype(jnp.float64)
            input_hi = input_hi.astype(jnp.float64)

        dtype = x_poly.c.dtype

        def _one(lo, hi, a_weight, a_coeff):
            res = crown_nn(Interval(lo, hi), a_weight, a_coeff)   # -> CROWNResult(lC,uC,ld,ud)
            # Use lC as the shared slope; ld/ud as lower/upper biases.
            T  = res.lC # [D, D]
            bl = res.ld # [D]
            bu = res.ud # [D]
            return T, bl, bu
        T, bl, bu = jax.vmap(_one)(input_lo, input_hi, affine_weight, affine_coeff)             # T:[B,D,D]
        T = T.astype(dtype)
        bl = bl.astype(dtype)
        bu = bu.astype(dtype)

        # Linear map
        c_vec = 0.5 * (bl + bu)                          # [B,D]
        r_vec = 0.5 * (bu - bl)                          # [B,D]
        return QuadPoly(c_vec, T, jnp.zeros_like(x_poly.L))

    return eval_poly

def build_nn_rhs_tm_aff(f: Callable, D: int, V: int):
    crown_nn = crown(f, in_len=V, out_len=D, enable_r=True)
    def eval_tm(x_tm: QuadTM, box_lo: Array, box_hi: Array) -> QuadTM:
        x_tm = x_tm.truncate_to_affine(box_lo, box_hi)
        affine_weight = x_tm.P.L
        affine_coeff  = x_tm.P.c
        R_lo = x_tm.R.lo
        R_hi = x_tm.R.hi
        input_lo = jnp.concatenate([box_lo, R_lo], axis=1) # [B,D_s * 2]
        input_hi = jnp.concatenate([box_hi, R_hi], axis=1) 
        # input_lo = jnp.concatenate([box_lo[:, 1:], jnp.zeros_like(R_lo)], axis=1) # [B,D_s * 2]
        # input_hi = jnp.concatenate([box_hi[:, 1:], jnp.zeros_like(R_hi)], axis=1) 
        if settings.CONFIG["FP64_IN_CROWN"]:
            affine_weight = affine_weight.astype(jnp.float64)
            affine_coeff  = affine_coeff.astype(jnp.float64)
            input_lo = input_lo.astype(jnp.float64)
            input_hi = input_hi.astype(jnp.float64)

        dtype = x_tm.P.c.dtype

        def _one(lo, hi, a_weight, a_coeff):
            res = crown_nn(Interval(lo, hi), a_weight, a_coeff)   # -> CROWNResult(lC,uC,ld,ud)
            # Use lC as the shared slope; ld/ud as lower/upper biases.
            T  = res.lC # [D, D]
            bl = res.ld # [D]
            bu = res.ud # [D]
            return T, bl, bu

        T, bl, bu = jax.vmap(_one)(input_lo, input_hi, affine_weight, affine_coeff)             # T:[B,D,D]
        T = T.astype(dtype)
        bl = bl.astype(dtype)
        bu = bu.astype(dtype)

        T_r = T[:, :, V:]                             # [B,D,D_r]
        T_r_pos = jnp.where(T_r>=0, T_r, jnp.zeros_like(T_r))
        T_r_neg = jnp.where(T_r<0, T_r, jnp.zeros_like(T_r))
        bl = bl + (T_r_pos @ R_lo[:,:,None] + T_r_neg @ R_hi[:,:,None]).squeeze(-1)  # [B,D]
        bu = bu + (T_r_pos @ R_hi[:,:,None] + T_r_neg @ R_lo[:,:,None]).squeeze(-1)  # [B,D]

        c_vec = 0.5 * (bl + bu)                          # [B,D]
        r_vec = 0.5 * (bu - bl)                          # [B,D]

        return QuadTM(QuadPoly(c_vec, T[:, :, :V], jnp.zeros_like(x_tm.P.L)), Interval(-r_vec, r_vec))

    return eval_tm


def build_auto_rhs_nn(f: Callable, D: int, V: int) -> Tuple[
    Callable[[QuadTM], QuadPoly],
    Callable[[QuadTM, Array, Array], QuadTM]
]:
    # return build_nn_rhs_poly(f, D, V), build_nn_rhs_tm(f, D, V)
    return build_nn_rhs_poly_aff(f, D, V), build_nn_rhs_tm_aff(f, D, V)
