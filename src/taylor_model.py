
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from src.interval import Interval, iv_sub_nonneg
from src.polynomial import QuadPoly, _poly_unary1, _poly_unary2
import src.settings as settings

Array = jnp.ndarray

# =========================
# Quadratic TM (poly + remainder)
# =========================
@dataclass
class QuadTM:
    """
    Shapes:
      P.c:[B,D], P.L:[B,D,V], P.Lt:[B,D,V];  R:[B,D]
    """
    P: QuadPoly
    R: Interval

    def __post_init__(self):
        if self.R.out_shape != self.P.out_shape:
            self.R = self.R.with_shape(self.P.out_shape)

    # ---- Basic ----
    @property
    def B(self): return self.P.B
    @property
    def D(self): return self.P.D
    @property
    def V(self): return self.P.V
    @property
    def logical_shape(self) -> tuple[int, ...]:
        return self.P.logical_shape

    def clone(self) -> "QuadTM":
        return QuadTM(self.P.clone(), self.R.clone())
    # ---- Constructors ----
    @staticmethod
    def zeros(B: int, D: int, V: int, dtype=jnp.float32) -> "QuadTM":
        return QuadTM(QuadPoly.zeros(B, D, V, dtype), Interval.zero(B, D, dtype))

    @staticmethod
    def const(B: int, D: int, V: int, c: float, dtype=jnp.float32) -> "QuadTM":
        return QuadTM(QuadPoly.const(B, D, V, c, dtype), Interval.zero(B, D, dtype))

    @staticmethod
    def var(B: int, D: int, V: int, idx: int, dtype=jnp.float32) -> "QuadTM":
        return QuadTM(QuadPoly.var(B, D, V, idx, dtype), Interval.zero(B, D, dtype))

    @staticmethod
    def from_poly(P: QuadPoly) -> "QuadTM":
        return QuadTM(P, Interval.zeros_like(P.c))

    # ======================
    # Helpers & coercions
    # ======================
    @staticmethod
    def to_tm_like(x, ref: QuadTM) -> QuadTM:
        """Coerce numeric or QuadPoly to QuadTM (shape Bx?xV) matching ref's batch & V."""
        if isinstance(x, QuadTM):
            return x
        if isinstance(x, QuadPoly):
            return QuadTM.from_poly(x)
        B = ref.P.c.shape[0]; V = ref.P.L.shape[2]
        c = jnp.asarray(x, ref.P.c.dtype)
        out_shape = None
        if c.ndim == 0:
            c = jnp.broadcast_to(c, (B, 1))
        elif c.ndim == 1:
            c = jnp.broadcast_to(c[None, :], (B, c.shape[0]))
        else:
            out_shape = c.shape
            c = jnp.broadcast_to(c.reshape((1, -1)), (B, c.size))
        L = jnp.zeros((B, c.shape[1], V), ref.P.L.dtype)
        Lt = jnp.zeros((B, c.shape[1], V), ref.P.L.dtype)
        return QuadTM.from_poly(QuadPoly(c, L, Lt, out_shape))

    def reshape(self, new_shape: tuple[int, ...]) -> "QuadTM":
        return QuadTM(self.P.reshape(new_shape), self.R.reshape(new_shape))

    def squeeze(self, axes: tuple[int, ...] | None = None) -> "QuadTM":
        return QuadTM(self.P.squeeze(axes), self.R.squeeze(axes))

    def broadcast_in_dim(self, shape: tuple[int, ...], broadcast_dimensions: tuple[int, ...]) -> "QuadTM":
        return QuadTM(
            self.P.broadcast_in_dim(shape, broadcast_dimensions),
            self.R.broadcast_in_dim(shape, broadcast_dimensions),
        )

    @staticmethod
    def slice(
        x: QuadTM,
        start: int | tuple[int, ...],
        limit: int | tuple[int, ...],
        strides: tuple[int, ...] | None = None,
    ) -> QuadTM:
        return QuadTM(QuadPoly.slice(x.P, start, limit, strides), x.R.slice(start, limit, strides))

    @staticmethod
    def concat(xs: list[QuadTM], axis: int = 0) -> QuadTM:
        P = QuadPoly.concat([x.P for x in xs], axis=axis)
        R = Interval.concat([x.R for x in xs], axis=axis)
        return QuadTM(P, R)

    def cumsum(self, axis: int | None = None) -> "QuadTM":
        """
        Prefix sum along the TM output/state dimension.
        Currently only supports 1D input semantics, i.e. axis=None or axis=1.
        """
        if axis not in (None, 1):
            raise NotImplementedError("QuadTM.cumsum currently only supports axis=None for 1D inputs")
        return QuadTM(self.P.cumsum(axis=axis), self.R.cumsum(axis=axis))

    def reduce_sum(self, axes: tuple[int, ...]) -> "QuadTM":
        return QuadTM(self.P.reduce_sum(axes), self.R.reduce_sum(axes))


    def append_zero_var(self, n: int) -> "QuadTM":
        return QuadTM(self.P.append_zero_var(n), self.R.append_zero_var(n))

    # ---- Algebra ----
    def add(self, other: "QuadTM") -> "QuadTM":
        return QuadTM(self.P.add(other.P),
                      Interval(self.R.lo + other.R.lo, self.R.hi + other.R.hi))

    def sub(self, other: "QuadTM") -> "QuadTM":
        return QuadTM(self.P.sub(other.P),
                      Interval(self.R.lo - other.R.hi, self.R.hi - other.R.lo))

    def scale(self, k: Array|float) -> "QuadTM":
        return QuadTM(self.P.scale(k), self.R.scale(k))

    # ---- Interval evaluation over a box ----
    def _eval_interval_affine(self, box_lo: Array, box_hi: Array) -> Interval:
        """
        Evaluate TM(z) = P(z) + R with P affine (Lt == 0).
        """
        I_poly = self.P._eval_interval_affine(box_lo, box_hi)
        return I_poly.add(self.R)

    def _eval_interval_naive(self, box_lo: Array, box_hi: Array) -> Interval:
        """
        Evaluate TM(z) = P(z) + R using naive interval eval for P.
        """
        Iv = self.P._eval_interval_naive(box_lo, box_hi)
        return Interval(Iv.lo + self.R.lo, Iv.hi + self.R.hi)

    def _eval_interval_horner(self, box_lo: Array, box_hi: Array) -> Interval:
        Iv = self.P._eval_interval_horner(box_lo, box_hi)
        return Interval(Iv.lo + self.R.lo, Iv.hi + self.R.hi)

    def _eval_interval_monimal(self, box_lo: Array, box_hi: Array) -> Interval:
        Iv = self.P._eval_interval_monimal(box_lo, box_hi)
        return Interval(Iv.lo + self.R.lo, Iv.hi + self.R.hi)

    def _eval_interval_quadbox(self, box_lo: Array, box_hi: Array) -> Interval:
        Iv = self.P._eval_interval_quadbox(box_lo, box_hi)
        return Interval(Iv.lo + self.R.lo, Iv.hi + self.R.hi)

    def eval_interval(self, box_lo: Array, box_hi: Array) -> Interval:
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            return self._eval_interval_affine(box_lo, box_hi)
        return self._eval_interval_horner(box_lo, box_hi)
        return self._eval_interval_quadbox(box_lo, box_hi)

    def truncate_to_affine(
        self,
        box_lo: Array,
        box_hi: Array,
    ) -> "QuadTM":
        """
        Truncate a QuadTM to an affine TM.
        Keeps P_aff (Lt dropped) and moves Lt's contribution into the remainder.

        Result:
        TM_out = (c + L·z) + (R + I_trunc) = (c + L'·z) + R,
        where I_trunc bounds ((Lt_t * t + (Lt_x·x)) * t) over the box. I_trunc is absorbed into L'.
        """
        P_aff = self.P.truncate_to_affine(box_lo, box_hi)
        return QuadTM(P_aff, self.R)
        # P_aff, I_trunc = self.P._truncate_to_affine(box_lo, box_hi)
        # R_out = self.R.add(I_trunc)
        # return QuadTM(P_aff, R_out)

    def _mul_ctrunc1(
        self,
        other: "QuadTM",
        box_lo: Array,
        box_hi: Array,
    ) -> "QuadTM":
        """
        TM×TM product with affine truncation on the polynomial part.
        Assumption: both TM.P are affine (Lt == 0); the result polynomial is affine.

        If A = (P1 + I1), B = (P2 + I2), then
        poly_out = affine part of (P1 P2)
        R_out    = I1*I2 + rng(P2)*I1 + rng(P1)*I2 + trunc_overflow(P1,P2)

        All interval ranges use the existing Horner-form evaluation.
        """
        P1, R1 = self.P, self.R
        P2, R2 = other.P, other.R

        # 1) Polynomial product (truncate to affine) + overflow of dropped quadratic
        P_out, I_trunc = P1._mul_ctrunc1(P2, box_lo, box_hi)

        # 2) Ranges of polynomial parts (no TM remainder)
        RP1 = P1._eval_interval_affine(box_lo, box_hi)  # rng(P1)
        RP2 = P2._eval_interval_affine(box_lo, box_hi)  # rng(P2)

        # 3) Mixed remainder terms
        I1xI2 = R1.mul(R2)
        P2xI1 = RP2.mul(R1)
        P1xI2 = RP1.mul(R2)

        R_out = Interval(
            I1xI2.lo + P2xI1.lo + P1xI2.lo + I_trunc.lo,
            I1xI2.hi + P2xI1.hi + P1xI2.hi + I_trunc.hi,
        )

        return QuadTM(P_out, R_out)

    def _mul_ctrunc2(self, other: "QuadTM",
                    box_lo: Array, box_hi: Array) -> "QuadTM":
        """
        Multiply two QuadTMs under the restricted quadratic basis:
            P(t,x) = c + L·z + t·(Lt·z),  z = [t, x1, ..., xD].
        We KEEP only monomials in {1, z_v, t·z_v} (including t^2 via v=0),
        and bound everything else into the interval remainder.

        Kept polynomial:
            kept_poly = self.P._mul_trunc2(other.P)

        New remainder:
            R_new = rng(P1)*R2 + rng(P2)*R1 + R1*R2 + overflow_poly,
        where overflow_poly collects all terms from P1*P2 that are NOT
        representable in {1, z_v, t·z_v}.
        """
        # 1) Polynomial-level: truncated product (project onto {1, z, t·z})
        kept_poly, poly_over = self.P.mul_ctrunc(other.P, box_lo, box_hi)

        # 2) Interval images of each polynomial (for mixed terms)
        I_left  = self.P.eval_interval(box_lo, box_hi)
        I_right = other.P.eval_interval(box_lo, box_hi)

        # 3) Mixed & remainder interactions (disjoint from poly_over)
        Rmix = (
            I_left.mul(other.R)
                .add(I_right.mul(self.R))
                .add(self.R.mul(other.R))
                .add(poly_over)
        )

        # 4) Assemble
        return QuadTM(kept_poly, Rmix)

    def mul(self, other: "QuadTM", box_lo: Array, box_hi: Array) -> "QuadTM":
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            return self._mul_ctrunc1(other, box_lo, box_hi)
        return self._mul_ctrunc2(other, box_lo, box_hi)

    def _recip_ctrunc1(self, box_lo: Array, box_hi: Array) -> "QuadTM":
        """
        Order-1 reciprocal of TM (Flow* style) under affine assumption (Lt==0).
        Keeps poly: (1/c) - (L·z)/c^2
        Remainder:   Horner multiply overflow (order-1) + Lagrange tail (p=2).
        """
        P, R = self.P, self.R
        c = P.c                              # [B,D]
        inv_c  = 1.0 / c
        # jax.debug.print("recip_ctrunc1: inv_c={}", inv_c)
        is_const_poly = self.P.is_const_poly()

        def _branch_const_poly():
            is_zero_R = self.R.is_zero()
            def _branch_zero_R():
                tm_out = self.clone()
                tm_out.P.c = inv_c
                return tm_out
            def _branch_nonzero_R():
                # Tight interval composition with recentering
                I_in  = R.add(c)   # c ⊕ R
                I_out = I_in.recip()
                # jax.debug.print("recip_ctrunc1: I_out={}", I_out)
                m = I_out.midpoint()
                tm_out = self.clone()
                tm_out.P.c = m
                tm_out.R = I_out.sub(m)
                return tm_out
            return jax.lax.cond(is_zero_R, _branch_zero_R, _branch_nonzero_R)
        
        def _branch_general_poly():
            return self._recip_ctrunc1_general(box_lo, box_hi)
        return jax.lax.cond(is_const_poly, _branch_const_poly, _branch_general_poly)

    def _recip_ctrunc1_general(self, box_lo: Array, box_hi: Array) -> "QuadTM":
        """
        Order-1 reciprocal of TM (Flow* style) under affine assumption (Lt==0).
        Keeps poly: (1/c) - (L·z)/c^2
        Remainder:   Horner multiply overflow (order-1) + Lagrange tail (p=2).
        """
        P, R = self.P, self.R
        c = P.c                              # [B,D]
        inv_c  = 1.0 / c

        # ---- Polynomial part: (1/c) - (L·z)/c^2 ----
        P_out = QuadPoly(
            c=inv_c,
            L=-P.L * (inv_c[:, :, None] * inv_c[:, :, None]),
            Lt=jnp.zeros_like(P.Lt),
        )

        # ---- Build G = (TM - c)/c (affine) ----
        # poly(G) = (L·z)/c,  rem(G) = R/c
        G_poly = QuadPoly(
            c=jnp.zeros_like(c),
            L=P.L * inv_c[:, :, None],
            Lt=jnp.zeros_like(P.Lt),
        )
        G = QuadTM(G_poly, R.scale(inv_c))

        # ================= Horner for (1 - G), order=1 =================
        # H := -1
        H = QuadTM.zeros(P.B, P.D, P.V)
        H.P.c = H.P.c - 1
        # H := H * G   (affine truncation captures multiply overflow like Flow*)
        H = H._mul_ctrunc1(G, box_lo, box_hi)
        # H.poly.c += 1
        H.P.c = H.P.c + 1
        # scale whole Horner block by 1/c
        H = H.scale(inv_c)

        # ===================== Lagrange tail (p = 2) ====================
        # I_F = range(F) = range(P - c) + R
        I_P   = P._eval_interval_affine(box_lo, box_hi)     # range(c + L·z)
        I_F   = I_P.sub(c).add(R)                       # (L·z)(box) + R
        Dom   = I_F.add(c)                              # c + I_F
        ratio = I_F.div(Dom)
        Tail  = ratio.pow(2).div(Dom).scale(inv_c)          # ((I_F/Dom)^2)/Dom * (1/c)

        # ---- Final remainder: Horner overflow + tail ----
        R_out = H.R.add(Tail)

        return QuadTM(P_out, R_out)

    def _div_ctrunc1(self, other: "QuadTM", box_lo: Array, box_hi: Array,) -> "QuadTM":
        """
        TM division with affine truncation (order-1).
        Assumptions:
        - self.P and other.P are affine (Lt == 0)
        - result polynomial is affine (Lt == 0)
        Implementation:
        self / other = self * (other^{-1})  with:
            - (other^{-1}) via QuadTM.recip_ctrunc1 (Lagrange/Flow* remainder)
            - product via QuadTM.mul_ctrunc_affine (adds trunc & mixed remainders)
        """
        recip_tm = other._recip_ctrunc1(box_lo, box_hi)     # TM for 1/other (order-1)
        return self._mul_ctrunc1(recip_tm, box_lo, box_hi)

    def _recip_ctrunc2(self, box_lo: Array, box_hi: Array) -> "QuadTM":
        """
        Order-2 reciprocal of TM (Flow* style).
        Keeps poly: (1/c) - (L·z)/c^2 + (L·z)^2/c^3
        Remainder:   Horner multiply overflow (order-2) + Lagrange tail (order 3).
        --------------------------------------------------------------------------
        Assumption: TM.P is quadratic (general Lt), result polynomial is quadratic (general Lt).
        --------------------------------------------------------------------------
        1/TM = 1/c  - (L·z)/c^2  + (L·z)^2/c^3  + Tail_3
        --------------------------------------------------------------------------
        Tail_3 := I_F^3 / (c + I_F)^4, where I_F is a sound interval enclosure of F on the box.
        Requires 0 ∉ (c + I_F).
        --------------------------------------------------------------------------
        """
        P, R = self.P, self.R
        c = P.c                              # [B,D]
        inv_c  = 1.0 / c
        # jax.debug.print("recip_ctrunc1: inv_c={}", inv_c)
        is_const_poly = self.P.is_const_poly()

        def _branch_const_poly():
            is_zero_R = self.R.is_zero()
            def _branch_zero_R():
                tm_out = self.clone()
                tm_out.P.c = inv_c
                return tm_out
            def _branch_nonzero_R():
                # Tight interval composition with recentering
                I_in  = R.add(c)   # c ⊕ R
                I_out = I_in.recip()
                # jax.debug.print("recip_ctrunc1: I_out={}", I_out)
                m = I_out.midpoint()
                tm_out = self.clone()
                tm_out.P.c = m
                tm_out.R = I_out.sub(m)
                return tm_out
            return jax.lax.cond(is_zero_R, _branch_zero_R, _branch_nonzero_R)
        
        def _branch_general_poly():
            return self._recip_ctrunc2_general(box_lo, box_hi)
        return jax.lax.cond(is_const_poly, _branch_const_poly, _branch_general_poly)

    def _recip_ctrunc2_general(self, box_lo: Array, box_hi: Array) -> "QuadTM":
        """
        Reciprocal TM with order-2 truncation on the restricted quadratic basis {1, z, t·z},
        plus a validated Lagrange remainder (order 3). Assumes t>=0 on the box.

        TM = c + F,   F = (L·z) + t(Lt·z) + R
        1/TM = 1/c  - F/c^2  + F^2/c^3  + Tail_3

        Tail_3 := I_F^3 / (c + I_F)^4, where I_F is a sound interval enclosure of F on the box.
        Requires 0 ∉ (c + I_F).
        """
        P, R = self.P, self.R
        c = P.c                           # [B,D]
        inv_c  = 1.0 / c
        inv_c2 = inv_c * inv_c
        inv_c3 = inv_c2 * inv_c

        # --------- Build E = TM - c (zero-constant TM) ---------
        E_poly = QuadPoly(c=jnp.zeros_like(c), L=P.L, Lt=P.Lt)
        E = QuadTM(E_poly, R)             # E = Δ1 + Δ2 + R

        # --------- A0: 1/c (poly only) ---------
        A0 = QuadTM(
            QuadPoly(c=inv_c,
                    L=jnp.zeros_like(P.L),
                    Lt=jnp.zeros_like(P.Lt)),
            Interval.zeros_like(R.lo)
        )

        # --------- A1: -(1/c^2) * E ---------
        A1 = E.scale(-inv_c2)

        # --------- A2: +(1/c^3) * (E * E) via mul_ctrunc2 ---------
        E2  = E._mul_ctrunc2(E, box_lo, box_hi)   # kept poly from Δ1^2 only; overflow collects others
        A2  = E2.scale(inv_c3)

        # --------- Partial sum (poly + rem) up to order 2 ---------
        PS_poly = QuadPoly(
            c  = A0.P.c  + A1.P.c  + A2.P.c,
            L  = A0.P.L  + A1.P.L  + A2.P.L,
            Lt = A0.P.Lt + A1.P.Lt + A2.P.Lt,
        )
        PS_rem = A0.R.add(A1.R).add(A2.R)

        # --------- Lagrange tail (order 3), with t>=0 Horner eval ---------
        # F(z) enclosure: eval polynomial P, subtract c, then add R
        I_P  = P.eval_interval(box_lo, box_hi)   # use your Horner t>=0 evaluation
        I_E  = I_P.sub(c).add(R)                 # = (Δ1+Δ2) + R
        D    = I_E.add(c)                        # = c + I_E   (must exclude 0)

        # Tail_3 = I_E^3 / D^4
        Tail3 = I_E.pow(3).div(D.pow(4))

        # --------- Final TM ---------
        R_out = PS_rem.add(Tail3)
        return QuadTM(PS_poly, R_out)

    def _div_ctrunc2(self, other: "QuadTM", box_lo: Array, box_hi: Array,) -> "QuadTM":
        """
        TM division with order-2 truncation on the restricted quadratic basis {1, z, t·z},
        plus a validated Lagrange remainder (order 3). Assumes t>=0 on the box.

        self / other = self * (other^{-1})  with:
            - (other^{-1}) via QuadTM.recip_ctrunc2 (Lagrange/Flow* remainder)
            - product via QuadTM.mul_ctrunc2 (adds trunc & mixed remainders)
        """
        recip_tm = other._recip_ctrunc2(box_lo, box_hi)     # TM for 1/other (order-2)
        return self._mul_ctrunc2(recip_tm, box_lo, box_hi)

    def recip(self, box_lo: Array, box_hi: Array) -> "QuadTM":
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            return self._recip_ctrunc1(box_lo, box_hi)
        return self._recip_ctrunc2(box_lo, box_hi)
    
    def div(self, other: "QuadTM", box_lo: Array, box_hi: Array,) -> "QuadTM":
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            return self._div_ctrunc1(other, box_lo, box_hi)
        return self._div_ctrunc2(other, box_lo, box_hi)

    def _integrate_time_ctrunc1(self, h: float, box_lo: Array, box_hi: Array) -> "QuadTM":
        """
        Indefinite time integration of a linear TM: TM = P + R  →  ∫TM dt.
        Assumptions:
        - INPUT polynomial P is affine (Lt == 0).
        - OUTPUT polynomial may have nonzero Lt (due to t^2 and t*x terms).
        - box_lo/box_hi provide the range of t (first coord of z).

        Output:
        P_out = self.P._integrate_time_trunc1()
        R_out = self.R * max(|t_lo|, |t_hi|)   (per-batch scalar, broadcast over D)
        """
        # 1) Polynomial part
        P_out = self.P._integrate_time_trunc1()

        # 2) Remainder: bound ∫ R dt over t∈[t_lo, t_hi] by ||R|| * max(|t_lo|,|t_hi|)
        t_lo = box_lo[:, 0:1]   # [B,1]
        t_hi = box_hi[:, 0:1]   # [B,1]
        t_mag = jnp.maximum(jnp.abs(t_lo), jnp.abs(t_hi))    # [B,1]

        R_out = self.R.scale(t_mag)

        return QuadTM(P_out, R_out)

    def _integrate_time_ctrunc2(self, h:float, box_lo: Array, box_hi: Array) -> "QuadTM":
        """
        Indefinite time integration of a general (order-2) TM: TM = P + R  →  ∫TM dt.
        Assumptions:
        - INPUT polynomial P has the partial quadratic form (c, L, Lt).
        - OUTPUT stays in the same basis; non-representable degree-3 terms from ∫P dt
        are captured in an interval overflow we add to the TM remainder.
        - t>=0 on the box (first coord of z).

        Output:
        P_out  = kept polynomial antiderivative (as in _integrate_time_trunc2)
        R_out  = (poly-integral overflow) + (R * t_max),  where t_max = max(|t_lo|,|t_hi|)=t_hi for t>=0
        """
        # 1) Polynomial: kept + overflow (degree-3) from integrating Lt-terms
        P_kept, Poly_over = self.P._integrate_time_ctrunc2(box_lo, box_hi)

        # 2) Integrate the existing TM remainder over t-interval
        t_lo = box_lo[:, 0:1]
        t_hi = box_hi[:, 0:1]
        t_mag = jnp.maximum(jnp.abs(t_lo), jnp.abs(t_hi))  # = t_hi if t>=0

        R_int = self.R.scale(t_mag)                        # ∫R dt ∈ R * t_max

        # 3) New TM remainder = polynomial-overflow + integrated remainder
        R_out = Poly_over.add(R_int)

        return QuadTM(P_kept, R_out)

    def integrate_time(self, h: float, box_lo: Array, box_hi: Array) -> "QuadTM":
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            return self._integrate_time_ctrunc1(h, box_lo, box_hi)
        return self._integrate_time_ctrunc2(h, box_lo, box_hi)

    def evaluate_time(self, h: float) -> "QuadTM":
        return QuadTM(self.P.evaluate_time(h), self.R)

    def is_zero(self) -> Array:
        """Check if TM is identically zero over all batches and dimensions."""
        return self.P.is_zero() & self.R.is_zero()

    def compose_affine(self, other: QuadTM, h: float) -> "QuadTM":
        """ other = A z + b + R_other
        self = c + L·z + t*(Lt·z) + R
        self.compose_affine(other) =
            c + L·(Az + b + R_other) + t*(Lt·(Az + b + R_other)) + R
        = c + L·(Az + b) + t*(Lt·(Az + b)) + (L + t*Lt)·R_other + R
        = P.compose_affine(other) + R_total
        R_total = R + (L + t*Lt)·R_other.
        """

        poly = self.P.compose_affine(other.P)
        # R_total = R + (L + t*Lt)·R_other
        R_total = self.R.add(other.R.affine(self.P.L[:, :, 1:]))
        R_total = R_total.add(other.R.affine(self.P.Lt[:, :, 1:]).scale(h))
        return QuadTM(poly, R_total)

    def log(self, prefix: str = "QuadTM", dim=None):
        jax.debug.print(f"------ {prefix} ------")
        self.P.log(dim=dim)
        self.R.log(dim=dim)
        return


def tm_unary_lagrange(
    A: QuadTM,
    f: Callable[[Array], Array],
    df: Callable[[Array], Array],
    fpp: Callable[[Array], Array],
    box_lo: Array,
    box_hi: Array,
    iv_f: Callable[[Interval], Interval],
    iv_fpp: Callable[[Interval], Interval],
    iv_fppp: Callable[[Interval], Interval],
) -> QuadTM:
    if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
        return _tm_unary_lagrange1(A, f, df, fpp, box_lo, box_hi, iv_f, iv_fpp, iv_fppp)
    else:
        return _tm_unary_lagrange2(A, f, df, fpp, box_lo, box_hi, iv_f, iv_fpp, iv_fppp)

def _tm_unary_lagrange1(
    A: QuadTM,
    f: Callable[[Array], Array],
    df: Callable[[Array], Array],
    fpp: Callable[[Array], Array],
    box_lo: Array,
    box_hi: Array,
    iv_f: Callable[[Interval], Interval],
    iv_fpp: Callable[[Interval], Interval],
    iv_fppp: Callable[[Interval], Interval],
) -> QuadTM:
    """
    Flow*-mirroring order-1 keep:
    P_keep = f(c) + f'(c) * g,  g = P - c.
    R_out  ⊇ f'(c) * R_in  ⊕  (1/2) * T^2 * hull(f''(c+T)),  T = range(g) ⊕ R_in.
    """

    c = A.P.c

    is_const_poly = A.P.is_const_poly()
    def _branch_const_poly(A: QuadTM):
        is_zero_R = A.R.is_zero()
        def _branch_zero_R(A: QuadTM):
            # Exact composition at a point
            tm_out = A.clone()
            tm_out.P.c = f(c)
            return tm_out
        def _branch_nonzero_R(A: QuadTM):
            # Tight interval composition with recentering
            I_in  = A.R.add(c)   # c ⊕ R
            I_out = iv_f(I_in)     # hull(f(c ⊕ R))
            m = I_out.midpoint()
            tm_out = A.clone()
            tm_out.P.c = m
            tm_out.R = I_out.sub(m)
            return tm_out
        return jax.lax.cond(is_zero_R, _branch_zero_R, _branch_nonzero_R, A)
    
    def _branch_general(A: QuadTM):
        return _tm_unary_lagrange1_general(A, f, df, box_lo, box_hi, iv_fpp)
    
    return jax.lax.cond(is_const_poly, _branch_const_poly, _branch_general, A)

def _tm_unary_lagrange1_general(
    A: QuadTM,
    f: Callable[[Array], Array],
    df: Callable[[Array], Array],
    box_lo: Array,
    box_hi: Array,
    iv_fpp: Callable[[Interval], Interval],
) -> QuadTM:
    """
    Flow*-mirroring order-1 keep:
    P_keep = f(c) + f'(c) * g,  g = P - c.
    R_out  ⊇ f'(c) * R_in  ⊕  (1/2) * T^2 * hull(f''(c+T)),  T = range(g) ⊕ R_in.
    """

    c = A.P.c
    # 1) Kept poly (affine)
    Pkeep = _poly_unary1(A.P, f, df)
    
    # 2) Linear propagation of input remainder: f'(c) * R_in
    s1 = df(A.P.c)                     # shape [B,D]
    R_lin = A.R.scale(s1)

    # 3) Range of g = P - c
    I_poly = A.P._eval_interval_affine(box_lo, box_hi)  # Interval of P
    
    G = I_poly.sub(c)    # I_poly - c

    # 4) T = G ⊕ R_in
    T = G.add(A.R)

    # 5) H = hull(f''(c + T))
    CplusT = T.add(c)
    H = iv_fpp(CplusT)

    # 6) R2 = 0.5 * T^2 * H
    T2 = T.square()
    half = Interval(jnp.asarray(0.5, A.P.c.dtype), jnp.asarray(0.5, A.P.c.dtype))
    R2 = T2.mul(H).mul(half)

    # 7) Total remainder
    R = R_lin.add(R2)
    return QuadTM(Pkeep, R)

def _tm_unary_lagrange2(
    A: QuadTM,
    f: Callable[[Array], Array],
    df: Callable[[Array], Array],
    fpp: Callable[[Array], Array],
    box_lo: Array,
    box_hi: Array,
    iv_f: Callable[[Interval], Interval],
    iv_fpp: Callable[[Interval], Interval],
    iv_fppp: Callable[[Interval], Interval],
) -> QuadTM:
    c = A.P.c

    is_const_poly = A.P.is_const_poly()
    def _branch_const_poly(A: QuadTM):
        is_zero_R = A.R.is_zero()
        def _branch_zero_R(A: QuadTM):
            # Exact composition at a point
            tm_out = A.clone()
            tm_out.P.c = f(c)
            return tm_out
        def _branch_nonzero_R(A: QuadTM):
            # Tight interval composition with recentering
            I_in  = A.R.add(c)   # c ⊕ R
            I_out = iv_f(I_in)     # hull(f(c ⊕ R))
            m = I_out.midpoint()
            tm_out = A.clone()
            tm_out.P.c = m
            tm_out.R = I_out.sub(m)
            return tm_out
        return jax.lax.cond(is_zero_R, _branch_zero_R, _branch_nonzero_R, A)
    
    def _branch_general(A: QuadTM):
        return _tm_unary_lagrange2_general(A, f, df, fpp, box_lo, box_hi, iv_fpp, iv_fppp)
    
    return jax.lax.cond(is_const_poly, _branch_const_poly, _branch_general, A)

def _tm_unary_lagrange2_general(
    A: QuadTM,
    f: Callable[[Array], Array],
    df: Callable[[Array], Array],
    fpp: Callable[[Array], Array],
    box_lo: Array,
    box_hi: Array,
    iv_fpp: Callable[[Interval], Interval],
    iv_fppp: Callable[[Interval], Interval],
) -> QuadTM:
    """
    Order-2 keep for unary f(TM) with restricted quadratic basis {1, z, t·z}, t>=0.
    Kept polynomial:
        P_keep = f(c)
               + f'(c) * (Δ1 + Δ2)
               + 0.5 * f''(c) * proj(Δ1^2)
      where Δ1 = L·z = a t + b·x, Δ2 = t(Lt·z) = u t^2 + Σ v_j t x_j,
            proj(Δ1^2) = a^2 t^2 + 2 a t (b·x).

    Remainder:
        R_out ⊇ f'(c) * R_in
                 ⊕ 0.5 * ( (T^2 ⊖ I_proj)_+ * hull(f''(c+T)) )
                 ⊕ (1/6) * T^3 * hull(f'''(c+T)),
      where T = range(Δ1+Δ2) ⊕ R_in, and
            I_proj = range( a^2 t^2 + 2 a t (b·x) )  (pure geometry).

    Notes:
    - Uses Horner eval for P to build ranges with t>=0.
    - Avoids double-counting the kept piece of Δ1^2 in the second-order remainder.
    """

    P, R = A.P, A.R
    c  = P.c                     # [B,D]
    L  = P.L                     # [B,D,V] (0: t, 1..: x)
    Lt = P.Lt                    # [B,D,V] (0: t^2, 1..: t x_j)
    B, Dm, V = L.shape
    assert V >= 2, "Expect at least one state dimension (V=D+1)."

    # ---- pointwise derivatives at c ----
    s0 = f(c)                    # [B,D]
    s1 = df(c)                   # [B,D]
    s2 = fpp(c)                  # [B,D]

    # ---- Kept polynomial ----
    a = L[:, :, 0]               # [B,D]      (L_t)
    b = L[:, :, 1:]              # [B,D,V-1]  (L_x)

    P_keep = _poly_unary2(A.P, f, df, fpp)

    # ---- Build T = range(Δ1+Δ2) ⊕ R_in  (full polynomial g = P - c) ----
    I_poly = P.eval_interval(box_lo, box_hi)        # Horner, t>=0
    G = I_poly.sub(c)                               # = range(Δ1+Δ2)
    T = G.add(R)                                    # include model remainder

    # ---- H = hull(f''(c + T)), H3 = hull(f'''(c + T)) ----
    CplusT = T.add(c)
    H  = iv_fpp(CplusT)
    H3 = iv_fppp(CplusT)

    # ---- I_proj = range( a^2 t^2 + 2 a t (b·x) ) on the box ----
    t_lo, t_hi = box_lo[:, 0], box_hi[:, 0]         # [B]
    Xlo,  Xhi  = box_lo[:, 1:], box_hi[:, 1:]       # [B,V-1]

    # a^2 t^2
    a2 = a * a                                      # [B,D] nonnegative
    t2_lo = (t_lo ** 2)[:, None]                    # [B,1], broadcast over D
    t2_hi = (t_hi ** 2)[:, None]
    a2t2_lo = jnp.minimum(a2 * t2_lo, a2 * t2_hi)   # [B,D]
    a2t2_hi = jnp.maximum(a2 * t2_lo, a2 * t2_hi)

    # 2 a t (b·x)
    # (b·x) interval via linear form
    Ix = QuadPoly._lin_form_interval(b, Xlo, Xhi)   # [B,D] interval of b·x
    It = Interval.from_scalar_bounds(t_lo, t_hi, Dm)  # [B,D] time interval
    tIx = It.mul(Ix)                                # [B,D] interval of t*(b·x)
    two_a = 2.0 * a
    two_a_tIx_lo = jnp.minimum(two_a * tIx.lo, two_a * tIx.hi)
    two_a_tIx_hi = jnp.maximum(two_a * tIx.lo, two_a * tIx.hi)

    I_proj = Interval(a2t2_lo + two_a_tIx_lo, a2t2_hi + two_a_tIx_hi)  # [B,D]

    # ---- Second-order remainder WITHOUT double-counting kept proj(Δ1^2) ----
    T2 = T.square()                               # [B,D]
    T2_rem = iv_sub_nonneg(T2, I_proj)     # remove the kept chunk
    half = Interval(jnp.array(0.5, c.dtype), jnp.array(0.5, c.dtype))
    R2 = T2_rem.mul(H).mul(half)

    # ---- Linear propagation & Lagrange 3rd-order tail ----
    R1 = R.scale(s1)                               # f'(c) * R_in
    T3 = T.pow(3)
    sixth = Interval(jnp.array(1.0/6.0, c.dtype), jnp.array(1.0/6.0, c.dtype))
    R3 = T3.mul(H3).mul(sixth)

    R_out = R1.add(R2).add(R3)
    return QuadTM(P_keep, R_out)


# ---------- PyTree registrations ----------
def _quadtm_flatten(tm: QuadTM):
    return ((tm.P, tm.R), None)
def _quadtm_unflatten(aux, children):
    P, R = children
    return QuadTM(P, R)

jax.tree_util.register_pytree_node(QuadTM, _quadtm_flatten, _quadtm_unflatten)
