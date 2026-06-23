
from __future__ import annotations
from dataclasses import dataclass
from math import prod
from typing import Tuple, Callable

import jax
import jax.numpy as jnp

from src.interval import Interval, iv_intersect
import src.settings as settings

Array = jnp.ndarray

# ========================================
# Quadratic (restricted) polynomial
#   P(t,x) = c + L·z + Lt·(t z)
# where z = [t, x1, ..., xD], Lt encodes only {t^2, t*xj} terms.
# ========================================
@dataclass
class QuadPoly:
    """
    Shapes:
      c  : [B, D]
      L  : [B, D, V]   (coeffs for z_v, with v=0 for time)
      Lt : [B, D, V]   (coeffs for t*z_v, i.e., t^2 (v=0) and t*x_j (v>=1))
    """
    c: Array
    L: Array
    Lt: Array
    out_shape: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.out_shape is not None:
            shape = tuple(int(s) for s in self.out_shape)
            if prod(shape) != self.c.shape[1]:
                raise ValueError(
                    f"QuadPoly out_shape={shape} incompatible with flat dim {self.c.shape[1]}"
                )
            self.out_shape = None if len(shape) <= 1 else shape

    # ---- Basic ----
    @property
    def B(self): return self.c.shape[0]
    @property
    def D(self): return self.c.shape[1]
    @property
    def V(self): return self.L.shape[2]
    @property
    def logical_shape(self) -> tuple[int, ...]:
        return self.out_shape if self.out_shape is not None else (self.D,)

    def clone(self) -> "QuadPoly":
        return QuadPoly(jnp.array(self.c, copy=True),
                        jnp.array(self.L, copy=True),
                        jnp.array(self.Lt, copy=True),
                        self.out_shape)

    # ---- Constructors ----
    @staticmethod
    def zeros(B: int, D: int, V: int, dtype=jnp.float32) -> "QuadPoly":
        zc = jnp.zeros((B, D), dtype)
        zV = jnp.zeros((B, D, V), dtype)
        return QuadPoly(zc, zV, zV)

    @staticmethod
    def const(B: int, D: int, V: int, c: float, dtype=jnp.float32) -> "QuadPoly":
        c0 = jnp.full((B, D), c, dtype)
        zV = jnp.zeros((B, D, V), dtype)
        return QuadPoly(c0, zV, zV)

    @staticmethod
    def var(B: int, D: int, V: int, idx: int, dtype=jnp.float32) -> "QuadPoly":
        L = jnp.zeros((B, D, V), dtype).at[:, :, idx].set(1.0)
        zc = jnp.zeros((B, D), dtype)
        Lt = jnp.zeros((B, D, V), dtype)
        return QuadPoly(zc, L, Lt)

    # ======================
    # Helpers & coercions
    # ======================

    @staticmethod
    def _normalize_shape(shape: tuple[int, ...] | None, flat_dim: int) -> tuple[int, ...] | None:
        if shape is None:
            return None
        shape = tuple(int(s) for s in shape)
        if prod(shape) != flat_dim:
            raise ValueError(f"Shape {shape} is incompatible with flat dim {flat_dim}")
        return None if len(shape) <= 1 else shape

    @staticmethod
    def _from_shaped(c: Array, L: Array, Lt: Array, out_shape: tuple[int, ...] | None) -> "QuadPoly":
        flat = int(prod(c.shape[1:])) if c.ndim > 1 else 1
        c_flat = c.reshape((c.shape[0], flat))
        L_flat = L.reshape((L.shape[0], flat, L.shape[-1]))
        Lt_flat = Lt.reshape((Lt.shape[0], flat, Lt.shape[-1]))
        return QuadPoly(c_flat, L_flat, Lt_flat, QuadPoly._normalize_shape(out_shape, flat))

    def _reshape_views(self) -> tuple[Array, Array, Array]:
        shape = self.logical_shape
        return (
            self.c.reshape((self.B, *shape)),
            self.L.reshape((self.B, *shape, self.V)),
            self.Lt.reshape((self.B, *shape, self.V)),
        )

    def with_shape(self, out_shape: tuple[int, ...] | None) -> "QuadPoly":
        return QuadPoly(self.c, self.L, self.Lt, out_shape)

    @staticmethod
    def to_poly_like(x, ref: QuadPoly) -> QuadPoly:
        """Coerce numeric or QuadTM to QuadPoly (shape Bx?xV) matching ref's batch & V."""
        if isinstance(x, QuadPoly):
            return x
        B = ref.c.shape[0]; V = ref.L.shape[2]
        c = jnp.asarray(x, ref.c.dtype)
        out_shape = None
        if c.ndim == 0:
            c = jnp.broadcast_to(c, (B, 1))
        elif c.ndim == 1:
            c = jnp.broadcast_to(c[None, :], (B, c.shape[0]))
        else:
            out_shape = c.shape
            c = jnp.broadcast_to(c.reshape((1, -1)), (B, c.size))
        L = jnp.zeros((B, c.shape[1], V), ref.L.dtype)
        Lt = jnp.zeros((B, c.shape[1], V), ref.L.dtype)
        return QuadPoly(c, L, Lt, QuadPoly._normalize_shape(out_shape, c.shape[1]))

    def reshape(self, new_shape: tuple[int, ...]) -> "QuadPoly":
        new_shape = tuple(int(s) for s in new_shape)
        if prod(new_shape) != self.D:
            raise ValueError(f"Cannot reshape QuadPoly of flat dim {self.D} to {new_shape}")
        return QuadPoly(self.c, self.L, self.Lt, QuadPoly._normalize_shape(new_shape, self.D))

    def squeeze(self, axes: tuple[int, ...] | None = None) -> "QuadPoly":
        shape = list(self.logical_shape)
        if axes is None:
            kept = tuple(dim for dim in shape if dim != 1)
        else:
            axes = tuple(int(ax) for ax in axes)
            rank = len(shape)
            norm_axes = tuple(ax if ax >= 0 else rank + ax for ax in axes)
            for ax in norm_axes:
                if shape[ax] != 1:
                    raise ValueError(f"Cannot squeeze axis {ax} of shape {tuple(shape)}")
            kept = tuple(dim for idx, dim in enumerate(shape) if idx not in norm_axes)
        kept = kept or (1,)
        return QuadPoly(self.c, self.L, self.Lt, QuadPoly._normalize_shape(kept, self.D))

    def broadcast_in_dim(self, shape: tuple[int, ...], broadcast_dimensions: tuple[int, ...]) -> "QuadPoly":
        shape = tuple(int(s) for s in shape)
        in_shape = self.logical_shape
        if len(broadcast_dimensions) == 0 and self.D == 1:
            scalar_shape = (self.B,) + (1,) * len(shape)
            c_exp = self.c[:, :1].reshape(scalar_shape)
            L_exp = self.L[:, :1, :].reshape(scalar_shape + (self.V,))
            Lt_exp = self.Lt[:, :1, :].reshape(scalar_shape + (self.V,))
            c_out = jnp.broadcast_to(c_exp, (self.B, *shape))
            L_out = jnp.broadcast_to(L_exp, (self.B, *shape, self.V))
            Lt_out = jnp.broadcast_to(Lt_exp, (self.B, *shape, self.V))
            return QuadPoly._from_shaped(c_out, L_out, Lt_out, shape)
        if len(in_shape) != len(broadcast_dimensions):
            raise ValueError(
                f"broadcast_dimensions={broadcast_dimensions} incompatible with input shape {in_shape}"
            )
        expanded = [1] * len(shape)
        for in_axis, out_axis in enumerate(broadcast_dimensions):
            expanded[int(out_axis)] = in_shape[in_axis]
        c_view, L_view, Lt_view = self._reshape_views()
        c_exp = c_view.reshape((self.B, *expanded))
        L_exp = L_view.reshape((self.B, *expanded, self.V))
        Lt_exp = Lt_view.reshape((self.B, *expanded, self.V))
        c_out = jnp.broadcast_to(c_exp, (self.B, *shape))
        L_out = jnp.broadcast_to(L_exp, (self.B, *shape, self.V))
        Lt_out = jnp.broadcast_to(Lt_exp, (self.B, *shape, self.V))
        return QuadPoly._from_shaped(c_out, L_out, Lt_out, shape)

    @staticmethod
    def slice(
        x: QuadPoly,
        start: int | tuple[int, ...],
        limit: int | tuple[int, ...],
        strides: tuple[int, ...] | None = None,
    ) -> QuadPoly:
        """Slice along the logical output shape, supporting rank-1 and rank-2 views."""
        if isinstance(start, tuple):
            starts = tuple(int(s) for s in start)
            limits = tuple(int(l) for l in limit)
            shape = x.logical_shape
            if strides is None:
                strides = (1,) * len(shape)
            elif len(strides) == 1 and len(shape) > 1:
                strides = tuple(int(strides[0]) for _ in shape)
            if not (len(starts) == len(limits) == len(strides) == len(shape)):
                raise ValueError(
                    f"Slice rank mismatch: starts={starts}, limits={limits}, strides={strides}, shape={shape}"
                )
            c_view, L_view, Lt_view = x._reshape_views()
            slicers = tuple(slice(s, l, st) for s, l, st in zip(starts, limits, strides))
            c_out = c_view[(slice(None),) + slicers]
            L_out = L_view[(slice(None),) + slicers + (slice(None),)]
            Lt_out = Lt_view[(slice(None),) + slicers + (slice(None),)]
            return QuadPoly._from_shaped(c_out, L_out, Lt_out, c_out.shape[1:])
        return QuadPoly(x.c[:, start:limit], x.L[:, start:limit, :], x.Lt[:, start:limit, :], x.out_shape)

    @staticmethod
    def concat(xs: list[QuadPoly], axis: int = 0) -> QuadPoly:
        if not xs:
            raise ValueError("QuadPoly.concat requires at least one input")
        shapes = [x.logical_shape for x in xs]
        rank = len(shapes[0])
        if any(len(shape) != rank for shape in shapes):
            raise ValueError(f"Cannot concatenate polynomials of different ranks: {shapes}")
        c_views = [x.c.reshape((x.B, *shape)) for x, shape in zip(xs, shapes)]
        L_views = [x.L.reshape((x.B, *shape, x.V)) for x, shape in zip(xs, shapes)]
        Lt_views = [x.Lt.reshape((x.B, *shape, x.V)) for x, shape in zip(xs, shapes)]
        c_out = jnp.concatenate(c_views, axis=axis + 1)
        L_out = jnp.concatenate(L_views, axis=axis + 1)
        Lt_out = jnp.concatenate(Lt_views, axis=axis + 1)
        return QuadPoly._from_shaped(c_out, L_out, Lt_out, c_out.shape[1:])

    def cumsum(self, axis: int | None = None) -> "QuadPoly":
        """
        Prefix sum along the polynomial output/state dimension.
        Currently only supports 1D input semantics, i.e. axis=None or axis=1.
        """
        if axis not in (None, 1):
            raise NotImplementedError("QuadPoly.cumsum currently only supports axis=None for 1D inputs")
        return QuadPoly(
            jnp.cumsum(self.c, axis=1),
            jnp.cumsum(self.L, axis=1),
            jnp.cumsum(self.Lt, axis=1),
            self.out_shape,
        )

    def reduce_sum(self, axes: tuple[int, ...]) -> "QuadPoly":
        shape = self.logical_shape
        rank = len(shape)
        norm_axes = tuple(ax if ax >= 0 else rank + ax for ax in axes)
        c_view, L_view, Lt_view = self._reshape_views()
        c_out = jnp.sum(c_view, axis=tuple(ax + 1 for ax in norm_axes))
        L_out = jnp.sum(L_view, axis=tuple(ax + 1 for ax in norm_axes))
        Lt_out = jnp.sum(Lt_view, axis=tuple(ax + 1 for ax in norm_axes))
        out_shape = tuple(dim for idx, dim in enumerate(shape) if idx not in norm_axes)
        return QuadPoly._from_shaped(c_out, L_out, Lt_out, out_shape or (1,))

    def append_zero_var(self, n) -> "QuadPoly":
        """Append a zero coefficient variable (increase D by n)."""
        B, D, V = self.B, self.D, self.V
        L_new  = jnp.concatenate([self.L,  jnp.zeros((B, n, V), dtype=self.L.dtype)], axis=1)
        Lt_new = jnp.concatenate([self.Lt, jnp.zeros((B, n, V), dtype=self.Lt.dtype)], axis=1)
        c_new  = jnp.concatenate([self.c,  jnp.zeros((B, n), dtype=self.c.dtype)], axis=1)
        return QuadPoly(c_new, L_new, Lt_new)

    # ---- Algebra ----
    def add(self, other: "QuadPoly") -> "QuadPoly":
        out_shape = self.out_shape if self.D >= other.D else other.out_shape
        return QuadPoly(self.c + other.c, self.L + other.L, self.Lt + other.Lt, out_shape)

    def sub(self, other: "QuadPoly") -> "QuadPoly":
        out_shape = self.out_shape if self.D >= other.D else other.out_shape
        return QuadPoly(self.c - other.c, self.L - other.L, self.Lt - other.Lt, out_shape)

    def scale(self, k: Array|float) -> "QuadPoly":
        if isinstance(k, jnp.ndarray) and k.ndim == 2:
            return QuadPoly(self.c * k, self.L * k[:, :, None], self.Lt * k[:, :, None], self.out_shape)
        return QuadPoly(self.c * k, self.L * k, self.Lt * k, self.out_shape)

    # ---- Interval evaluation over box [lo,hi] in z-space ----
    @staticmethod
    def _lin_form_interval(a: Array, lo: Array, hi: Array) -> Interval:
        """
        Interval of a · z over z∈[lo,hi].  Shapes: a:[B,D,N], lo/hi:[B,N] → [B,D].
        """
        lo_e = lo[:, None, :]
        hi_e = hi[:, None, :]
        t_lo = jnp.minimum(a * lo_e, a * hi_e).sum(axis=-1)
        t_hi = jnp.maximum(a * lo_e, a * hi_e).sum(axis=-1)
        return Interval(t_lo, t_hi)

    def _eval_interval_affine(self, box_lo: Array, box_hi: Array) -> Interval:
        """
        Evaluate P(z) = c + L·z on z∈[box_lo, box_hi].
        Assumption: affine mode (Lt == 0).
        """
        # interval of L·z over the full box (first dim is t, rest are x’s)
        I_lin = QuadPoly._lin_form_interval(self.L, box_lo, box_hi)  # [B,D]
        return Interval(self.c + I_lin.lo, self.c + I_lin.hi)

    def _eval_interval_naive(self, box_lo: Array, box_hi: Array) -> Interval:
        """
        Interval of P(t,x)=c + L·z + t*(Lt·z), where z=[t,x].
        """
        B, D, V = self.B, self.D, self.V
        # linear part L·z
        Lz = QuadPoly._lin_form_interval(self.L, box_lo, box_hi)      # [B,D] interval
        # time-linear part: t * (Lt·z)
        It = Interval.from_scalar_bounds(box_lo[:, 0], box_hi[:, 0], D)
        Lt_z = QuadPoly._lin_form_interval(self.Lt, box_lo, box_hi)   # [B,D] interval for (Lt·z)
        tz = It.mul(Lt_z)                                             # [B,D] interval for t*(Lt·z)
        return Interval(self.c + Lz.lo + tz.lo, self.c + Lz.hi + tz.hi)

    def _eval_interval_horner(self, box_lo: Array, box_hi: Array) -> Interval:
        """
        Evaluate P(t,x) = c + L·z + t*(Lt·z) over z=[t,x]
        using Horner-form interval evaluation to reduce dependency loss.

        Rewrites P as:
            P(t,x) = ((Lt_t * t + (L_t + Lt_x·x)) * t) + (c + L_x·x)
        """
        B, D, V = self.B, self.D, self.V
        t_lo, t_hi = box_lo[:, 0], box_hi[:, 0]

        # ---- Split L and Lt into time and spatial parts ----
        L_t  = self.L[:, :, 0:1]   # coeff of t
        L_x  = self.L[:, :, 1:]    # coeffs of x
        Lt_t = self.Lt[:, :, 0:1]  # coeff of t^2
        Lt_x = self.Lt[:, :, 1:]   # coeffs of t*x_j

        # ---- Build component intervals ----
        # Spatial contributions
        Ix   = QuadPoly._lin_form_interval(L_x,  box_lo[:, 1:], box_hi[:, 1:])   # L_x·x
        ILtx = QuadPoly._lin_form_interval(Lt_x, box_lo[:, 1:], box_hi[:, 1:])   # Lt_x·x

        # Time intervals
        It = Interval.from_scalar_bounds(t_lo, t_hi, D)

        # ---- Horner-style evaluation ----
        # Inner term: (Lt_t * t) + (L_t + Lt_x·x)
        inner = Interval(Lt_t.squeeze(-1), Lt_t.squeeze(-1)).mul(It).add(
            Interval(L_t.squeeze(-1), L_t.squeeze(-1)).add(ILtx)
        )

        # Outer term: t * inner + (c + L_x·x)
        base  = Interval(self.c + Ix.lo, self.c + Ix.hi)
        outer = It.mul(inner).add(base)

        return outer

    def _eval_interval_monimal(self, box_lo: Array, box_hi: Array) -> Interval:
        """
        Tight interval evaluation for
            P(t,x) = c + L_t * t + (L_x · x) + (Lt_t * t^2) + t * (Lt_x · x),
        where:
            L_t  = L[:,:,0]      (coeff of t)
            L_x  = L[:,:,1:]     (coeffs of x)
            Lt_t = Lt[:,:,0]     (coeff of t^2 via t*(Lt·z) picking z_0=t)
            Lt_x = Lt[:,:,1:]    (coeffs of t*x_j via t*(Lt·z) picking z_j=x_j)
        This avoids putting 't' in both factors of a single interval product.
        """
        B, D, V = self.B, self.D, self.V
        # Split coefficients
        L_t  = self.L[:, :, 0]     # (B,D)
        L_x  = self.L[:, :, 1:]    # (B,D,V-1)
        Lt_t = self.Lt[:, :, 0]    # (B,D)
        Lt_x = self.Lt[:, :, 1:]   # (B,D,V-1)

        # Box pieces
        t_lo, t_hi = box_lo[:, 0], box_hi[:, 0]           # (B,)
        Xlo,  Xhi  = box_lo[:, 1:], box_hi[:, 1:]         # (B,V-1)

        # 1) Linear-in-x term: L_x · x
        Ix = QuadPoly._lin_form_interval(L_x, Xlo, Xhi)   # Interval (B,D)

        # 2) Linear-in-x for Lt_x: (Lt_x · x)
        I_tx = QuadPoly._lin_form_interval(Lt_x, Xlo, Xhi)  # Interval (B,D)

        # 3) Time interval
        It = Interval.from_scalar_bounds(t_lo, t_hi, D)   # Interval (B,D)

        # 4) Square-bounds for t^2 (lower=0 if [t_lo,t_hi] straddles 0)
        tlo2 = jnp.minimum(t_lo * t_lo, t_hi * t_hi)
        thi2 = jnp.maximum(t_lo * t_lo, t_hi * t_hi)
        tlo2 = jnp.where((t_lo <= 0) & (t_hi >= 0), jnp.zeros_like(tlo2), tlo2)
        # Broadcast to (B,D)
        tlo2 = tlo2[:, None]
        thi2 = thi2[:, None]
        It2 = Interval(tlo2, thi2)                        # Interval (B,D)

        # 5) Assemble each monomial group with sign-aware interval products
        # base: c + (L_x · x)
        I_base = Interval(self.c + Ix.lo, self.c + Ix.hi)

        # L_t * t
        I_Lt_t = Interval(L_t, L_t).mul(It)

        # Lt_t * t^2
        I_t2 = Interval(Lt_t, Lt_t).mul(It2)

        # t * (Lt_x · x)
        I_bilin = It.mul(I_tx)

        # Total
        return I_base.add(I_Lt_t).add(I_t2).add(I_bilin)

    def _eval_interval_quadbox(self, box_lo: Array, box_hi: Array) -> Interval:
        """
        Exact min/max of P(t,x)=c + a*t + b·x + u*t^2 + Σ_j v_j * t * x_j
        over the axis-aligned box t∈[t_lo,t_hi], x_j∈[xlo_j,xhi_j].

        Strategy (batch-vectorized):
        • For each x-corner (2^(V-1) of them), q_k(t) is a 1D quadratic:
                q_k(t) = u*t^2 + (a + (v·x_k))*t + (c + (b·x_k)).
            Its extrema on [t_lo,t_hi] occur at clamped vertex and/or endpoints.
        • The global min/max over the rectangle is attained among these
            candidates across all x-corners.

        Shapes:
        c,a,u : [B, D]
        b,v   : [B, D, S] with S = V-1 (spatial dims)
        box_lo/hi: [B, V]
        Returns Interval with lo/hi: [B, D]
        """
        B, D, V = self.B, self.D, self.V
        assert V >= 1
        S = V - 1  # #spatial vars
        t_lo = box_lo[:, 0]  # [B]
        t_hi = box_hi[:, 0]  # [B]
        Xlo, Xhi = box_lo[:, 1:], box_hi[:, 1:]  # [B, S] (S may be 0)

        # Split coefficients
        c = self.c                   # [B, D]
        a = self.L[:, :, 0]          # [B, D]  (L_t)
        u = self.Lt[:, :, 0]         # [B, D]  (Lt_t)
        b = self.L[:, :, 1:]         # [B, D, S]
        v = self.Lt[:, :, 1:]        # [B, D, S]

        # Handle S==0 (no spatial variables) as a fast path
        if S == 0:
            # q(t) = u t^2 + a t + c
            tlo = t_lo[:, None, None]  # [B,1,1]
            thi = t_hi[:, None, None]
            uBD = u[..., None]         # [B,D,1]
            aBD = a[..., None]
            cBD = c[..., None]
            # vertex
            eps = jnp.asarray(1e-18, dtype=self.c.dtype)
            has_vertex = (jnp.abs(uBD) > eps)
            t_star = -aBD / (2.0 * uBD + jnp.where(has_vertex, 0.0, 1.0))  # safe
            t_star = jnp.clip(jnp.where(has_vertex, t_star, tlo), tlo, thi)

            def q_at(t):
                return uBD * (t * t) + aBD * t + cBD  # [B,D,1]

            q_candidates = jnp.concatenate([q_at(tlo), q_at(thi), q_at(t_star)], axis=-1)  # [B,D,3]
            lo = jnp.min(q_candidates, axis=-1)  # [B,D]
            hi = jnp.max(q_candidates, axis=-1)
            return Interval(lo, hi)

        # Build all 2^S binary corner selectors for x
        C = 1 << S
        # bits: [C, S] in {0,1}
        bits = (jnp.arange(C, dtype=jnp.uint32)[:, None] >> jnp.arange(S, dtype=jnp.uint32)) & 1
        bits = bits.astype(self.c.dtype)  # float for arithmetic

        # X corners per batch: [B, C, S]
        Xc = Xlo[:, None, :] * (1.0 - bits[None, :, :]) + Xhi[:, None, :] * bits[None, :, :]

        # Compute (b·x_c) and (v·x_c) for every corner: [B, D, C]
        # einsum: (B,D,S)·(B,C,S) -> (B,D,C)
        bx = jnp.einsum('bds,bcs->bdc', b, Xc)  # [B,D,C]
        vx = jnp.einsum('bds,bcs->bdc', v, Xc)  # [B,D,C]

        # Coefficients of q_k(t) per corner k
        uBDC = u[..., None]          # [B,D,1]
        aBDC = a[..., None]          # [B,D,1]
        cBDC = c[..., None]          # [B,D,1]
        lin  = aBDC + vx             # [B,D,C]
        con  = cBDC + bx             # [B,D,C]

        # Evaluate candidates at t_lo, t_hi, and vertex (when u≠0)
        tlo = t_lo[:, None, None]    # [B,1,1]
        thi = t_hi[:, None, None]    # [B,1,1]

        def q_at(t):                 # t: [B,1,1] or [B,D,C]
            return uBDC * (t * t) + lin * t + con  # broadcasting

        q_tlo = q_at(tlo)            # [B,D,C]
        q_thi = q_at(thi)            # [B,D,C]

        eps = jnp.asarray(1e-18, dtype=self.c.dtype)
        has_vertex = (jnp.abs(uBDC) > eps)        # [B,D,1]
        t_star = -lin / (2.0 * uBDC + jnp.where(has_vertex, 0.0, 1.0))  # [B,D,C], safe divide
        # clamp vertices to [t_lo,t_hi]
        t_star = jnp.clip(t_star, tlo, thi)       # [B,D,C]
        # If no vertex (u≈0), just reuse t_lo (any choice inside interval is fine)
        t_star = jnp.where(has_vertex, t_star, jnp.broadcast_to(tlo, t_star.shape))

        q_tstar = q_at(t_star)       # [B,D,C]

        # Min/max over corners and candidates
        q_min = jnp.minimum(jnp.minimum(q_tlo, q_thi), q_tstar)  # [B,D,C]
        q_max = jnp.maximum(jnp.maximum(q_tlo, q_thi), q_tstar)  # [B,D,C]

        lo = jnp.min(q_min, axis=-1)  # [B,D]
        hi = jnp.max(q_max, axis=-1)
        return Interval(lo, hi)

    def eval_interval(self, box_lo: Array, box_hi: Array) -> Interval:
        """
        Evaluate polynomial over box [box_lo, box_hi] in z-space.
        method: "naive", "horner", "monimal", or "quadbox"
        """
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            return self._eval_interval_affine(box_lo, box_hi)
        return self._eval_interval_horner(box_lo, box_hi)
        return self._eval_interval_quadbox(box_lo, box_hi)

    def _truncate_to_affine(
        self,
        box_lo: Array,
        box_hi: Array,
    ) -> Tuple["QuadPoly", Interval]:
        """
        Truncate to affine: keep (c, L) and drop Lt.
        Overflow is bounded directly as ((Lt_t * t + (Lt_x·x)) * t).

        Returns:
        P_aff  : affine-only polynomial (Lt == 0)
        I_trunc: interval bound of dropped Lt contribution over the box
        """
        B, D, V = self.B, self.D, self.V

        # 1) Keep affine part
        P_aff = QuadPoly(
            c=self.c,
            L=self.L,
            Lt=jnp.zeros_like(self.Lt),
        )

        # 2) Bound overflow from Lt: ((Lt_t * t) + (Lt_x · x)) * t
        # Build interval for t
        t_lo, t_hi = box_lo[:, 0], box_hi[:, 0]                 # [B]
        It = Interval.from_scalar_bounds(t_lo, t_hi, D)         # [B,D] interval

        # Split Lt into (t, x)
        Lt_t = self.Lt[:, :, 0:1]                               # [B,D,1]
        Lt_x = self.Lt[:, :, 1:]                                # [B,D,V-1] (may be empty if V==1)

        # Lt_t * t  (coeff interval is exact: [a,a])
        ItLt_t = Interval(Lt_t.squeeze(-1), Lt_t.squeeze(-1)).mul(It)  # [B,D]

        # (Lt_x · x)
        if V > 1:
            Ix = QuadPoly._lin_form_interval(Lt_x, box_lo[:, 1:], box_hi[:, 1:])  # [B,D]
        else:
            Ix = Interval.zeros_like(self.c)

        # Sum inside parentheses, then multiply by t
        inner = ItLt_t.add(Ix)          # [B,D]
        I_trunc = inner.mul(It)         # [B,D]

        return P_aff, I_trunc

    def truncate_to_affine(
        self,
        box_lo: Array,
        box_hi: Array,
    ) -> QuadPoly:
        
        P_aff, I_trunc = self._truncate_to_affine(box_lo, box_hi)

        c_offset = I_trunc.midpoint()
        o_radius = I_trunc.radius()
        i_radius = (box_hi - box_lo)[:, 1:] * 0.5 # [B,V]
        L_offset = jnp.eye(self.D, self.V, k=1, dtype=P_aff.L.dtype)[None] * (o_radius / (i_radius + 1e-12))[:, :, None]
        L_offset = jnp.where(jnp.positive(P_aff.L), L_offset, -L_offset)

        P_aff.c = P_aff.c + c_offset
        P_aff.L = P_aff.L + L_offset

        return P_aff

    def _mul_trunc1(self, other: "QuadPoly") -> "QuadPoly":
        """
        Affine×Affine -> Affine polynomial product (truncate quadratic).
        Assumption: inputs are affine (Lt == 0); output is affine (Lt == 0).

        If P1 = c1 + L1·z, P2 = c2 + L2·z, then
        keep:   c  = c1*c2
                L  = c1*L2 + c2*L1
        drop:   (L1·z)(L2·z)  (sent to overflow by mul_ctrunc_affine).
        """
        c1, L1 = self.c, self.L
        c2, L2 = other.c, other.L

        c_out = c1 * c2
        L_out = L1 * c2[:, :, None] + L2 * c1[:, :, None]
        Lt_out = jnp.zeros_like(self.Lt)  # keep affine only

        return QuadPoly(c_out, L_out, Lt_out, self.out_shape)

    def _mul_ctrunc1(self, other: "QuadPoly",
                        box_lo: Array, box_hi: Array) -> Tuple["QuadPoly", Interval]:
        """
        Order-1 keep: keep affine; overflow = full quadratic of (L1·z)(L2·z).
        Diagonal z_i^2 via square bounds (nonnegative lower); off-diagonal z_i z_j via 4-corner bounds.
        """
        kept = self._mul_trunc1(other)  # c=c1*c2, L=c1*L2 + c2*L1, Lt=0

        a1 = self.L    # (B,D,V)
        a2 = other.L   # (B,D,V)
        Zlo, Zhi = box_lo, box_hi           # (B,V)
        B, D, V = self.B, self.D, self.V

        # --- helpers ---
        def square_bounds(l, u):
            lo2 = jnp.minimum(l*l, u*u)
            hi2 = jnp.maximum(l*l, u*u)
            lo2 = jnp.where((l <= 0) & (u >= 0), jnp.zeros_like(lo2), lo2)
            return lo2, hi2

        # Diagonal z_i^2 terms: Cdiag = a1 * a2 (elementwise on last axis)
        Cdiag = a1 * a2                         # (B,D,V)
        Z2_lo, Z2_hi = square_bounds(Zlo, Zhi)  # (B,V)
        lo_diag = jnp.minimum(Cdiag * Z2_lo[:, None, :],
                            Cdiag * Z2_hi[:, None, :]).sum(axis=2)        # (B,D)
        hi_diag = jnp.maximum(Cdiag * Z2_lo[:, None, :],
                            Cdiag * Z2_hi[:, None, :]).sum(axis=2)

        # Off-diagonal z_i z_j terms via 4-corner bounds
        # Coefficient tensor C_{ij} = a1_i * a2_j
        C = a1[:, :, :, None] * a2[:, :, None, :]        # (B,D,V,V)

        Zlo_i = Zlo[:, :, None]  # (B,V,1)
        Zhi_i = Zhi[:, :, None]
        Zlo_j = Zlo[:, None, :]  # (B,1,V)
        Zhi_j = Zhi[:, None, :]

        q1 = Zlo_i * Zlo_j
        q2 = Zlo_i * Zhi_j
        q3 = Zhi_i * Zlo_j
        q4 = Zhi_i * Zhi_j
        ZZ_lo = jnp.minimum(jnp.minimum(q1, q2), jnp.minimum(q3, q4))  # (B,V,V)
        ZZ_hi = jnp.maximum(jnp.maximum(q1, q2), jnp.maximum(q3, q4))  # (B,V,V)

        # Mask off the diagonal (already accrued in lo_diag/hi_diag)
        eye = jnp.eye(V, dtype=ZZ_lo.dtype)[None, :, :]   # (1,V,V)
        mask_off = 1.0 - eye
        ZZ_lo_off = ZZ_lo * mask_off
        ZZ_hi_off = ZZ_hi * mask_off

        lo_off = jnp.minimum(C * ZZ_lo_off[:, None, :, :],
                            C * ZZ_hi_off[:, None, :, :]).sum(axis=(2, 3))  # (B,D)
        hi_off = jnp.maximum(C * ZZ_lo_off[:, None, :, :],
                            C * ZZ_hi_off[:, None, :, :]).sum(axis=(2, 3))

        lo = lo_diag + lo_off
        hi = hi_diag + hi_off
        over = Interval(lo, hi)
        return kept, over

    def _mul_trunc2(self, other: "QuadPoly") -> "QuadPoly":
        """
        Truncated product of two QuadPolys onto the restricted basis {1, z_v, t·z_v}.

        Let
        P1 = c1 + L1·z + t·(Lt1·z),   P2 = c2 + L2·z + t·(Lt2·z),  z=[t,x1,...,xD].
        Expand P1*P2 and KEEP ONLY the monomials in {1, z_v, t·z_v}:
        - constant:      c = c1*c2
        - linear in z:   L = c1*L2 + c2*L1
        - linear in (t·z): Lt =
                c1*Lt2 + c2*Lt1
            + (L1_t * L2) + (L2_t * L1) - (L1_t * L2_t)·e0

        Notes:
        • Lk_t means the time column of Lk (the coefficient of z_0 = t).
        • The last subtraction DOES NOT remove the t^2 term. It CORRECTS
            double counting of t^2 that arises when expanding (L1·z)(L2·z)
            into t·z_v contributions: the v=0 case (t·t) would appear twice;
            subtracting one copy yields the single correct t^2 coefficient.
        """
        # constant
        c = self.c * other.c

        # linear (z)
        L = self.c[:, :, None] * other.L + other.c[:, :, None] * self.L

        # time-linear (t·z)
        Lt = (
            self.c[:, :, None] * other.Lt +
            other.c[:, :, None] * self.Lt +
            self.L[:, :, 0:1] * other.L +
            other.L[:, :, 0:1] * self.L
        )
        # subtract only on v=0
        Lt = Lt.at[:, :, 0].add(- (self.L[:, :, 0] * other.L[:, :, 0]))


        return QuadPoly(c, L, Lt, self.out_shape)

    def mul(self, other: "QuadPoly") -> "QuadPoly":
        """
        Truncated product of two QuadPolys onto the restricted basis {1, z_v, t·z_v}.
        Wrapper for _mul_trunc2.
        """
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            return self._mul_trunc1(other)
        return self._mul_trunc2(other)

    def _mul_ctrunc2(self, other: "QuadPoly",
                    box_lo: Array, box_hi: Array) -> Tuple["QuadPoly", Interval]:
        """
        Full polynomial product projected onto {1, z_v, t·z_v}, plus overflow bounds.

        Overflow collects all non-representable terms from P1*P2, namely:
        (A) pure spatial quadratic:             Σ_{i,j≥1} L1_i L2_j x_i x_j
        (B) cubic terms with one t:             t·(L1·z)(Lt2·z) + t·(L2·z)(Lt1·z)
        (C) quartic terms with t^2:             t^2·(Lt1·z)(Lt2·z)

        We bound (A) with exact bilinear convex hull over axis-aligned boxes:
        - diagonals x_i^2 via square-bounds (nonnegativity accounted),
        - off-diagonals x_ix_j (i≠j) via 4-corner min/max.
        We bound (B) and (C) by interval products of their factors.
        """
        kept = self._mul_trunc2(other)

        B, D, V = self.B, self.D, self.V
        Zlo, Zhi = box_lo, box_hi
        Xlo, Xhi = Zlo[:, 1:], Zhi[:, 1:]        # spatial box

        # ----- helpers -----
        def square_bounds(l, u):
            # Tight bounds for x^2 on [l,u]
            lo2 = jnp.minimum(l*l, u*u)
            hi2 = jnp.maximum(l*l, u*u)
            # If interval straddles 0, min is 0
            lo2 = jnp.where((l <= 0) & (u >= 0), jnp.zeros_like(lo2), lo2)
            return lo2, hi2

        # ===== (A) pure spatial (L1·x)(L2·x) =====
        a1_sp, a2_sp = self.L[:, :, 1:], other.L[:, :, 1:]  # (B, D, V-1)

        # Diagonal: Σ_k (L1_xk * L2_xk) * x_k^2
        Cdiag   = a1_sp * a2_sp                             # (B, D, V-1)
        X2_lo, X2_hi = square_bounds(Xlo, Xhi)              # (B, V-1)
        lo_diag = jnp.minimum(Cdiag * X2_lo[:, None, :], Cdiag * X2_hi[:, None, :]).sum(-1)  # (B, D)
        hi_diag = jnp.maximum(Cdiag * X2_lo[:, None, :], Cdiag * X2_hi[:, None, :]).sum(-1)  # (B, D)

        # Off-diagonal: Σ_{i≠j} (L1_xi * L2_xj) * x_i x_j  via 4-corner bounds of the bilinear form
        # Build all pairs (i,j)
        C = a1_sp[:, :, :, None] * a2_sp[:, :, None, :]     # (B, D, V-1, V-1)

        Xlo_i, Xhi_i = Xlo[:, :, None], Xhi[:, :, None]     # (B, V-1, 1)
        Xlo_j, Xhi_j = Xlo[:, None, :], Xhi[:, None, :]     # (B, 1, V-1)

        q1 = Xlo_i * Xlo_j
        q2 = Xlo_i * Xhi_j
        q3 = Xhi_i * Xlo_j
        q4 = Xhi_i * Xhi_j

        XX_lo = jnp.minimum(jnp.minimum(q1, q2), jnp.minimum(q3, q4))  # (B, V-1, V-1)
        XX_hi = jnp.maximum(jnp.maximum(q1, q2), jnp.maximum(q3, q4))

        # mask out diagonal (i=j): already accounted in diagonal term
        eye = jnp.eye(V-1, dtype=XX_lo.dtype)[None, :, :]   # (1, V-1, V-1)
        XX_lo_off = XX_lo * (1.0 - eye)
        XX_hi_off = XX_hi * (1.0 - eye)

        lo_off = jnp.minimum(C * XX_lo_off[:, None, :, :], C * XX_hi_off[:, None, :, :]).sum((2, 3))  # (B, D)
        hi_off = jnp.maximum(C * XX_lo_off[:, None, :, :], C * XX_hi_off[:, None, :, :]).sum((2, 3))  # (B, D)

        over_xx = Interval(lo_diag + lo_off, hi_diag + hi_off)  # (B, D) interval

        # ===== (B) cubic terms with one t: t·(L·z)(Lt·z) =====
        L1z  = QuadPoly._lin_form_interval(self.L,  Zlo, Zhi)   # rng(L1·z) over full z-box
        L2z  = QuadPoly._lin_form_interval(other.L, Zlo, Zhi)
        Lt1z = QuadPoly._lin_form_interval(self.Lt, Zlo, Zhi)   # rng(Lt1·z)
        Lt2z = QuadPoly._lin_form_interval(other.Lt, Zlo, Zhi)

        It   = Interval.from_scalar_bounds(Zlo[:, 0], Zhi[:, 0], D)  # t in [t_lo, t_hi], shape (B,D)

        tL1Lt2 = It.mul(L1z.mul(Lt2z))   # t*(L1·z)(Lt2·z)
        tL2Lt1 = It.mul(L2z.mul(Lt1z))   # t*(L2·z)(Lt1·z)
        over_tz2 = tL1Lt2.add(tL2Lt1)

        # ===== (C) quartic with t^2: t^2·(Lt1·z)(Lt2·z) =====
        over_t2z2 = It.mul(It).mul(Lt1z.mul(Lt2z))

        # ===== Sum all overflow components =====
        over = Interval(
            over_xx.lo + over_tz2.lo + over_t2z2.lo,
            over_xx.hi + over_tz2.hi + over_t2z2.hi
        )

        return kept, over

    def _mul_ctrunc2_tight(self, other: "QuadPoly",
                    box_lo: Array, box_hi: Array) -> Tuple["QuadPoly", Interval]:
        """
        Full polynomial product projected onto {1, z_v, t·z_v}, plus a *tightened* overflow.

        Overflow collects non-representable terms from P1*P2:
        (A) (L1_x·x)(L2_x·x)          [pure spatial quadratic]
        (B) t·(L1·z)(Lt2·z) + sym     [cubic in (t,x)]
        (C) t^2·(Lt1·z)(Lt2·z)        [quartic in (t,x)]

        Tightening trick (t >= 0):
        For (B) and (C), compute two safe intervals and take their intersection:
            - “prod”  = interval-of-products form (your original)
            - “mono”  = monomial-by-monomial sum with exact square/4-corner for x-terms
        """
        kept = self._mul_trunc2(other)

        B, D, V = self.B, self.D, self.V
        Zlo, Zhi = box_lo, box_hi
        Xlo, Xhi = Zlo[:, 1:], Zhi[:, 1:]                 # (B, V-1)
        t_lo, t_hi = Zlo[:, 0], Zhi[:, 0]                 # (B,)

        # ---------- helpers ----------
        def square_bounds(l, u):
            lo2 = jnp.minimum(l*l, u*u)
            hi2 = jnp.maximum(l*l, u*u)
            lo2 = jnp.where((l <= 0) & (u >= 0), jnp.zeros_like(lo2), lo2)
            return lo2, hi2

        def bilinear_box_bounds(xlo, xhi):
            # For each pair (i,j), bound x_i x_j via 4-corner min/max
            q1 = xlo[:, :, None] * xlo[:, None, :]
            q2 = xlo[:, :, None] * xhi[:, None, :]
            q3 = xhi[:, :, None] * xlo[:, None, :]
            q4 = xhi[:, :, None] * xhi[:, None, :]
            XX_lo = jnp.minimum(jnp.minimum(q1, q2), jnp.minimum(q3, q4))  # (B, V-1, V-1)
            XX_hi = jnp.maximum(jnp.maximum(q1, q2), jnp.maximum(q3, q4))
            return XX_lo, XX_hi

        def mul_coef_interval_sum(coef, Llo, Lhi, axis):
            """
            Elementwise interval product of 'coef' (can be negative) with [Llo, Lhi],
            then sum over 'axis'. Shapes broadcast.
            """
            p1 = coef * Llo
            p2 = coef * Lhi
            lo = jnp.minimum(p1, p2).sum(axis=axis)
            hi = jnp.maximum(p1, p2).sum(axis=axis)
            return lo, hi

        # ---------- (A) pure spatial (L1_x·x)(L2_x·x) ----------
        a1_x, a2_x = self.L[:, :, 1:], other.L[:, :, 1:]            # (B, D, V-1)
        X2_lo, X2_hi = square_bounds(Xlo, Xhi)                      # (B, V-1)

        # diagonal x_k^2
        Cdiag = a1_x * a2_x                                         # (B, D, V-1)
        lo_diag, hi_diag = mul_coef_interval_sum(Cdiag, X2_lo[:, None, :],
                                                X2_hi[:, None, :], axis=-1)

        # off-diagonal x_i x_j via 4-corner
        XX_lo, XX_hi = bilinear_box_bounds(Xlo, Xhi)                # (B, V-1, V-1)
        eye = jnp.eye(V-1, dtype=XX_lo.dtype)[None, :, :]
        XX_lo_off = XX_lo * (1.0 - eye)
        XX_hi_off = XX_hi * (1.0 - eye)

        # coefficients for off-diagonals: sum_{i != j} (a1_xi * a2_xj) x_i x_j
        Cmat = a1_x[:, :, :, None] * a2_x[:, :, None, :]            # (B, D, V-1, V-1)
        # Interval product and sum over (i,j)
        p1 = Cmat * XX_lo_off[:, None, :, :]
        p2 = Cmat * XX_hi_off[:, None, :, :]
        lo_off = jnp.minimum(p1, p2).sum(axis=(2, 3))
        hi_off = jnp.maximum(p1, p2).sum(axis=(2, 3))

        over_A = Interval(lo_diag + lo_off, hi_diag + hi_off)       # (B, D)

        # ---------- Precompute linear-form intervals ----------
        L1z  = QuadPoly._lin_form_interval(self.L,  Zlo, Zhi)       # (B, D)
        L2z  = QuadPoly._lin_form_interval(other.L, Zlo, Zhi)
        Lt1z = QuadPoly._lin_form_interval(self.Lt, Zlo, Zhi)       # (B, D)
        Lt2z = QuadPoly._lin_form_interval(other.Lt, Zlo, Zhi)

        It   = Interval.from_scalar_bounds(t_lo, t_hi, D)           # (B, D), t>=0 assumed
        It2  = It.square()             # (B, D)
        It3  = It.pow(3)             # (B, D)
        It4  = It.pow(4)             # (B, D)

        # ---------- (B) cubic with one t ----------
        # (B_prod) = It * ( rng(L1·z)*rng(Lt2·z) + rng(L2·z)*rng(Lt1·z) )
        B_prod = It.mul(L1z.mul(Lt2z).add(L2z.mul(Lt1z)))

        # (B_mono): monomial sum with exact x-bounds
        # Decompose coefficients:
        a1_t = self.L[:, :, 0]            # (B, D)
        a2_t = other.L[:, :, 0]
        u1_t = self.Lt[:, :, 0]
        u2_t = other.Lt[:, :, 0]
        b1_x = self.L[:, :, 1:]           # (B, D, V-1)
        b2_x = other.L[:, :, 1:]
        v1_x = self.Lt[:, :, 1:]          # (B, D, V-1)
        v2_x = other.Lt[:, :, 1:]

        # t^3 term: (a1_t*u2_t + a2_t*u1_t) * t^3
        coef_t3 = a1_t * u2_t + a2_t * u1_t
        B_t3 = Interval(coef_t3, coef_t3).mul(It3)

        # t^2 x_j terms: sum_j ( (a1_t*v2_j + b1_j*u2_t) + (a2_t*v1_j + b2_j*u1_t) ) * t^2 * x_j
        coef_t2x = (a1_t[:, :, None] * v2_x + b1_x * u2_t[:, :, None] +
                    a2_t[:, :, None] * v1_x + b2_x * u1_t[:, :, None])   # (B, D, V-1)
        # t^2 * x_j interval: It2 (B,D) times Ix_j (B,1) -> broadcast to (B,D,V-1)
        Ix_lo, Ix_hi = Xlo, Xhi                                             # (B, V-1)
        t2x_lo = It2.lo[:, :, None] * jnp.where(Ix_lo <= Ix_hi, Ix_lo, Ix_hi)[:, None, :]  # (B,D,V-1)
        t2x_hi = It2.hi[:, :, None] * jnp.where(Ix_lo <= Ix_hi, Ix_hi, Ix_lo)[:, None, :]
        lo_t2x, hi_t2x = mul_coef_interval_sum(coef_t2x, t2x_lo, t2x_hi, axis=-1)
        B_t2x = Interval(lo_t2x, hi_t2x)

        # t * x_i x_j terms: sum_{i,j} (b1_i*v2_j + b2_i*v1_j) * t * (x_i x_j)
        coef_tx2 = (b1_x[:, :, :, None] * v2_x[:, :, None, :] +
                    b2_x[:, :, :, None] * v1_x[:, :, None, :])            # (B, D, V-1, V-1)
        # x_i x_j bounds
        XX_lo, XX_hi = bilinear_box_bounds(Xlo, Xhi)                       # (B, V-1, V-1)
        # include diagonal (i=j) automatically; bilinear_box_bounds already gives exact x^2 and 4-corner
        # multiply by t interval
        tXX_lo = It.lo[:, :, None, None] * XX_lo[:, None, :, :]
        tXX_hi = It.hi[:, :, None, None] * XX_hi[:, None, :, :]
        lo_tx2 = jnp.minimum(coef_tx2 * tXX_lo, coef_tx2 * tXX_hi).sum(axis=(2, 3))
        hi_tx2 = jnp.maximum(coef_tx2 * tXX_lo, coef_tx2 * tXX_hi).sum(axis=(2, 3))
        B_tx2 = Interval(lo_tx2, hi_tx2)

        B_mono = B_t3.add(B_t2x).add(B_tx2)
        over_B = iv_intersect(B_prod, B_mono)

        # ---------- (C) quartic with t^2 ----------
        # (C_prod) = It^2 * ( rng(Lt1·z)*rng(Lt2·z) )
        C_prod = It2.mul(Lt1z.mul(Lt2z))

        # (C_mono): t^4 + t^3 x + t^2 x_i x_j with exact x-bounds
        # t^4: (u1_t*u2_t)*t^4
        coef_t4 = u1_t * u2_t
        C_t4 = It4.scale(coef_t4)

        # t^3 x_j: sum_j (u1_t*v2_j + u2_t*v1_j) * t^3 * x_j
        coef_t3x = u1_t[:, :, None] * v2_x + u2_t[:, :, None] * v1_x        # (B,D,V-1)
        t3x_lo = It3.lo[:, :, None] * jnp.where(Ix_lo <= Ix_hi, Ix_lo, Ix_hi)[:, None, :]
        t3x_hi = It3.hi[:, :, None] * jnp.where(Ix_lo <= Ix_hi, Ix_hi, Ix_lo)[:, None, :]
        lo_t3x, hi_t3x = mul_coef_interval_sum(coef_t3x, t3x_lo, t3x_hi, axis=-1)
        C_t3x = Interval(lo_t3x, hi_t3x)

        # t^2 x_i x_j: sum_{i,j} (v1_i*v2_j) * t^2 * (x_i x_j)
        coef_t2x2 = v1_x[:, :, :, None] * v2_x[:, :, None, :]               # (B,D,V-1,V-1)
        t2XX_lo = It2.lo[:, :, None, None] * XX_lo[:, None, :, :]
        t2XX_hi = It2.hi[:, :, None, None] * XX_hi[:, None, :, :]
        lo_t2x2 = jnp.minimum(coef_t2x2 * t2XX_lo, coef_t2x2 * t2XX_hi).sum(axis=(2, 3))
        hi_t2x2 = jnp.maximum(coef_t2x2 * t2XX_lo, coef_t2x2 * t2XX_hi).sum(axis=(2, 3))
        C_t2x2 = Interval(lo_t2x2, hi_t2x2)

        C_mono = C_t4.add(C_t3x).add(C_t2x2)
        over_C = iv_intersect(C_prod, C_mono)

        # ---------- sum overflow ----------
        over = Interval(
            over_A.lo + over_B.lo + over_C.lo,
            over_A.hi + over_B.hi + over_C.hi
        )
        return kept, over

    def mul_ctrunc(self, other: "QuadPoly",
                    box_lo: Array, box_hi: Array) -> Tuple["QuadPoly", Interval]:
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            return self._mul_ctrunc1(other, box_lo, box_hi)
        return self._mul_ctrunc2(other, box_lo, box_hi)

    def _recip_trunc1(self) -> "QuadPoly":
        """
        Affine reciprocal, polynomial part only (order-1).
        Assumption: self is affine (Lt == 0); output is affine (Lt == 0).

        For P = c + L·z  →  (1/P)_{≤1} = (1/c) - (L·z)/c^2.
        """
        c = self.c                     # [B,D]
        L = self.L                     # [B,D,V]
        inv_c  = 1.0 / c               # [B,D]
        inv_c2 = inv_c * inv_c         # [B,D]

        c_out  = inv_c                 # 1/c
        L_out  = -L * inv_c2[:, :, None]
        Lt_out = jnp.zeros_like(self.Lt)
        return QuadPoly(c_out, L_out, Lt_out, self.out_shape)

    def _div_trunc1(self, other: "QuadPoly") -> "QuadPoly":
        """
        Affine division, polynomial part only (order-1).
        Assumptions:
        - self and other are affine (Lt == 0)
        - output polynomial is affine (Lt == 0)
        Implementation:
        (self / other)_{≤1} = self * (other^{-1})_{≤1}
        """
        recip_poly = other._recip_trunc1()          # (1/other)_{≤1}
        return self._mul_trunc1(recip_poly)   # keep affine part

    def _recip_trunc2(self) -> "QuadPoly":
        """
        Order-2 reciprocal on the restricted quadratic basis {1, z, t·z}.
        Keeps polynomial terms that fit the basis; higher/non-representable
        parts (e.g., (b·x)^2, 2Δ1Δ2, Δ2^2) are *not* included here.

        Math:
        P = c + L·z + t(Lt·z), with
            a = L[..., 0]      (coeff of t)
            b = L[..., 1:]     (coeffs of x_j)
            u = Lt[..., 0]     (coeff of t^2)
            v = Lt[..., 1:]    (coeffs of t x_j)

        (1/P)_{≤2} = 1/c
                    - (L·z)/c^2
                    - t(Lt·z)/c^2
                    + (a^2 t^2 + 2 a t (b·x)) / c^3

        Returns:
        QuadPoly(c_out, L_out, Lt_out)
        """
        c  = self.c                  # [B,D]
        L  = self.L                  # [B,D,V]  (V=D+1, index 0 is t)
        Lt = self.Lt                 # [B,D,V]

        inv_c  = 1.0 / c             # [B,D]
        inv_c2 = inv_c * inv_c       # [B,D]
        inv_c3 = inv_c2 * inv_c      # [B,D]

        # Split time vs spatial parts
        a = L[:, :, 0]               # [B,D]      (L_t)
        b = L[:, :, 1:]              # [B,D,V-1]  (L_x)
        u = Lt[:, :, 0]              # [B,D]      (Lt_t)
        v = Lt[:, :, 1:]             # [B,D,V-1]  (Lt_x)

        # Constant
        c_out = inv_c

        # Linear in z
        L_out = -L * inv_c2[:, :, None]

        # t·z part:
        #   Lt_out[..., 0]   (t^2):  -u/c^2 + (a^2)/c^3
        #   Lt_out[..., 1:] (t xj): -v/c^2 + (2 a b_j)/c^3
        Lt_out = jnp.empty_like(Lt)
        Lt_out = Lt_out.at[:, :, 0].set(-u * inv_c2 + (a * a) * inv_c3)
        Lt_out = Lt_out.at[:, :, 1:].set(-v * inv_c2[:, :, None] + (2.0 * a)[:, :, None] * b * inv_c3[:, :, None])

        return QuadPoly(c_out, L_out, Lt_out, self.out_shape)

    def _div_trunc2(self, other: "QuadPoly") -> "QuadPoly":
        """
        Order-2 division on the restricted quadratic basis {1, z, t·z}.
        Keeps polynomial terms that fit the basis; higher/non-representable
        parts (e.g., (b·x)^2, 2Δ1Δ2, Δ2^2) are *not* included here.

        Implementation:
        (self / other)_{≤2} = self * (other^{-1})_{≤2}
        """
        recip_poly = other._recip_trunc2()          # (1/other)_{≤2}
        return self._mul_trunc2(recip_poly)         # keep order-2 part

    def recip(self) -> "QuadPoly":
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            return self._recip_trunc1()
        return self._recip_trunc2()

    def div(self, other: "QuadPoly") -> "QuadPoly":
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            return self._div_trunc1(other)
        return self._div_trunc2(other)

    def _integrate_time_trunc1(self) -> "QuadPoly":
        """
        Indefinite integral wrt time t, keeping the output in our (c, L, Lt) basis.
        Assumptions:
        - INPUT is affine (Lt == 0).
        - OUTPUT may have nonzero Lt (because integration creates t^2 and t*x).
        Implements:
        ∫(c + L_t t + L_x·x) dt
            = (c) t + 0.5*(L_t) t^2 + (L_x·x) t
        Map to basis: c'=0,
            L_t'  = c,
            L_x'  = 0,
            Lt_t' = 0.5 * L_t,
            Lt_x' = L_x.
        """
        c  = self.c                  # [B,D]
        L  = self.L                  # [B,D,V]; V>=1 with z=[t,x...]
        Lt = self.Lt                 # [B,D,V], assumed zero but we won't assert

        B, D, V = L.shape

        # c' = 0
        c_out = jnp.zeros_like(c)

        # L' : only time slot (index 0) gets c; spatial slots become 0
        L_out = jnp.zeros_like(L)
        L_out = L_out.at[:, :, 0].set(c)            # L_t' = c

        # Lt' : time slot gets 0.5 * L_t; spatial slots copy L_x
        Lt_out = jnp.zeros_like(Lt)
        Lt_out = Lt_out.at[:, :, 0].set(0.5 * L[:, :, 0])   # Lt_t' = 0.5*L_t
        if V > 1:
            Lt_out = Lt_out.at[:, :, 1:].set(L[:, :, 1:])   # Lt_x' = L_x

        return QuadPoly(c_out, L_out, Lt_out, self.out_shape)

    def _integrate_time_trunc2(self) -> "QuadPoly":
        c  = self.c
        L  = self.L
        Lt = self.Lt
        # ---- Kept polynomial (same mapping as affine integration) ----
        # c' = 0
        c_out = jnp.zeros_like(c)

        # L' : only time slot gets c; spatial slots 0
        L_out = jnp.zeros_like(L)
        L_out = L_out.at[:, :, 0].set(c)                  # L'_t = c

        # Lt' : time slot gets 0.5*a; spatial slots copy b
        a = L[:, :, 0]                                    # [B,D]
        b = L[:, :, 1:]                                   # [B,D,V-1]
        Lt_out = jnp.zeros_like(Lt)
        Lt_out = Lt_out.at[:, :, 0].set(0.5 * a)          # Lt'_t = 0.5 * a
        Lt_out = Lt_out.at[:, :, 1:].set(b)               # Lt'_x = b

        kept = QuadPoly(c_out, L_out, Lt_out, self.out_shape)
        return kept

    def _integrate_time_ctrunc1(self, box_lo: Array, box_hi: Array) -> Tuple["QuadPoly", Interval]:
        """
        Indefinite integral wrt time t for an affine P=c + L·z.
        Keeps only representable terms in {1, z, t·z}; returns (kept_poly, overflow_interval).

        ∫P dt = c t + 0.5 a t^2 + (b·x) t
                └──────── kept ───────┘ 
        Shapes:
        c:  [B,D]
        L:  [B,D,V]   with index 0 = t, 1..V-1 = x_j
        box_*: [B,V]
        """
        c  = self.c
        L  = self.L
        # ---- Kept polynomial: c t + 0.5 a t^2 + (b·x) t ----
        kept = self._integrate_time_trunc1()

        # ---- Overflow interval: none for affine input ----
        overflow = Interval.zeros_like(c)
        return kept, overflow

    def _integrate_time_ctrunc2(self, box_lo: Array, box_hi: Array) -> Tuple["QuadPoly", Interval]:
        """
        Indefinite integral wrt time t for a partial-quadratic P=c + L·z + t(Lt·z).
        Keeps only representable terms in {1, z, t·z}; returns (kept_poly, overflow_interval).

        ∫P dt = c t + 0.5 a t^2 + (b·x) t   +   (1/3) u t^3 + 0.5 Σ_j v_j t^2 x_j
                └──────── kept ───────┘         └────────── overflow (degree-3) ─────────┘

        Shapes:
        c:  [B,D]
        L:  [B,D,V]   with index 0 = t, 1..V-1 = x_j
        Lt: [B,D,V]   with index 0 = t^2, 1..V-1 = t*x_j
        box_*: [B,V]
        """
        c  = self.c
        L  = self.L
        Lt = self.Lt
        # ---- Kept polynomial: c t + 0.5 a t^2 + (b·x) t ----
        kept = self._integrate_time_trunc2()

        # ---- Overflow interval: (1/3) u t^3 + 0.5 Σ_j v_j t^2 x_j ----
        t_lo, t_hi = box_lo[:, 0], box_hi[:, 0]           # [B]
        Xlo,  Xhi  = box_lo[:, 1:], box_hi[:, 1:]         # [B,V-1]

        u = Lt[:, :, 0]                                   # [B,D]
        v = Lt[:, :, 1:]                                  # [B,D,V-1]

        # t^3 interval (t>=0): [t_lo^3, t_hi^3], broadcast to [B,D]
        t3_lo = (t_lo ** 3)[:, None]
        t3_hi = (t_hi ** 3)[:, None]
        # u * t^3 : elementwise interval product
        ut3_lo = jnp.minimum(u * t3_lo, u * t3_hi)
        ut3_hi = jnp.maximum(u * t3_lo, u * t3_hi)
        term_t3 = Interval((1.0/3.0) * ut3_lo, (1.0/3.0) * ut3_hi)

        # For each j: t^2 in [t_lo^2, t_hi^2], x_j in [Xlo_j, Xhi_j].
        t2_lo = (t_lo ** 2)[:, None, None]                # [B,1,1]
        t2_hi = (t_hi ** 2)[:, None, None]                # [B,1,1]
        x_lo  = Xlo[:, None, :]                           # [B,1,V-1]
        x_hi  = Xhi[:, None, :]                           # [B,1,V-1]

        # Interval for t^2 * x_j (four endpoints)
        p1 = t2_lo * x_lo
        p2 = t2_lo * x_hi
        p3 = t2_hi * x_lo
        p4 = t2_hi * x_hi
        t2x_lo = jnp.minimum(jnp.minimum(p1, p2), jnp.minimum(p3, p4))   # [B,1,V-1]
        t2x_hi = jnp.maximum(jnp.maximum(p1, p2), jnp.maximum(p3, p4))   # [B,1,V-1]

        # Multiply by v_j and sum over j
        pv_lo = jnp.minimum(v * t2x_lo, v * t2x_hi).sum(axis=-1)         # [B,D]
        pv_hi = jnp.maximum(v * t2x_lo, v * t2x_hi).sum(axis=-1)         # [B,D]
        term_t2x = Interval(0.5 * pv_lo, 0.5 * pv_hi)

        overflow = Interval(term_t3.lo + term_t2x.lo,
                            term_t3.hi + term_t2x.hi)
        return kept, overflow

    def integrate_time_trunc(self) -> "QuadPoly":
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            return self._integrate_time_trunc1()
        return self._integrate_time_trunc2()

    def integrate_time_ctrunc(self, box_lo: Array, box_hi: Array) -> Tuple["QuadPoly", Interval]:
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            raise NotImplementedError("Affine integration with overflow not implemented.")
        return self._integrate_time_ctrunc2(box_lo, box_hi)

    # ---- Time substitution t := h, folding to degree-1 form ----
    def evaluate_time(self, h: float) -> "QuadPoly":
        """
        Substitute t:=h, folding t^2 and t*x_j into constants/linear-x.
        Returns a QuadPoly with Lt=0 and L[:, :, 0]=0 (degree-1 in z).
        """
        # c' = c + h*L_t + h^2*Lt_t
        c_new = self.c + self.L[:, :, 0] * h + self.Lt[:, :, 0] * (h * h)
        # Lx' = L_x + h * Lt_x
        Lx_new = self.L[:, :, 1:] + self.Lt[:, :, 1:] * h
        # Assemble L' with time axis zeroed
        L_new = jnp.zeros_like(self.L).at[:, :, 1:].set(Lx_new)
        Lt_new = jnp.zeros_like(self.Lt)
        return QuadPoly(c_new, L_new, Lt_new, self.out_shape)

    def is_zero(self) -> bool:
        """Check if polynomial is identically zero over all batches and dimensions."""
        return jnp.allclose(self.c, 0) & jnp.allclose(self.L, 0) & jnp.allclose(self.Lt, 0)

    def is_const_poly(self) -> Array:
        """Check if polynomial is identically zero poly over all batches and dimensions."""
        return jnp.allclose(self.L, 0) & jnp.allclose(self.Lt, 0)

    # ---- Affine composition z ↦ A z + b, with time at index 0 ----
    def compose_affine(self, other: QuadPoly) -> "QuadPoly":
        """
        A:[B,D,V], b:[B,D] ->
        A:[B,V,V], b:[B,V].
        For P=c + L·z + t*(Lt·z), under z -> Az + b and keeping t as first coord:
          P = c + L·(Az + b) + t*(Lt·(Az + b))
        so:
          L'  = L @ A
          c'  = c + (L · b)
          Lt' = Lt @ A
          L'[:, :, 0] += (Lt · b)     # because t*(Lt·b) contributes to the t slot
        """
        A = other.L      # [B,D,V]
        b = other.c      # [B,D]
        t_id = jnp.zeros((self.B, 1, self.V)).at[:, :, 0].set(1.0)
        A = jnp.concat([t_id, A], axis=1)  # [B,V,V]
        b = jnp.concat([jnp.zeros((self.B, 1)), b], axis=1)  # [B,V]
        L_new  = jnp.einsum("bdv,bvw->bdw", self.L,  A)   # [B,D,V]
        c_new  = self.c + (self.L * b[:, None, :]).sum(axis=-1)
        Lt_new = jnp.einsum("bdv,bvw->bdw", self.Lt, A)   # [B,D,V]
        # Add t*(Lt·b) to L[:, :, 0]
        t_extra = (self.Lt * b[:, None, :]).sum(axis=-1)  # [B,D]
        L_new = L_new.at[:, :, 0].add(t_extra)
        return QuadPoly(c_new, L_new, Lt_new, self.out_shape)

    def log(self, prefix: str = "QuadPoly", dim=None):
        if dim is None:
            jax.debug.print(f"{prefix}: c={self.c.tolist()}, L={self.L.tolist()}, Lt={self.Lt.tolist()}")
        else:
            jax.debug.print(f"{prefix}: c={self.c[:, dim].tolist()}, L={self.L[:, dim].tolist()}, Lt={self.Lt[:, dim].tolist()}")
        return

# ======================
# Poly-level nonpoly helpers
# ======================
def poly_unary(P: QuadPoly,
                 f: Callable[[Array], Array],
                 df: Callable[[Array], Array],
                 fpp: Callable[[Array], Array]) -> QuadPoly:
    if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
        return _poly_unary1(P, f, df, fpp)
    else:
        return _poly_unary2(P, f, df, fpp)

def _poly_unary1(P: QuadPoly, f: Callable[[Array], Array], df: Callable[[Array], Array], *args) -> QuadPoly:
    """Apply unary non-polynomial at poly level: 1st-order around constants; scale both L and Lt."""
    c = f(P.c)
    s = df(P.c)  # [B,D]
    L  = P.L  * s[:, :, None]
    Lt = P.Lt * s[:, :, None]
    return QuadPoly(c, L, Lt, P.out_shape)

def _poly_unary2(P: QuadPoly,
                 f: Callable[[Array], Array],
                 df: Callable[[Array], Array],
                 fpp: Callable[[Array], Array]) -> QuadPoly:
    """Apply unary non-polynomial at poly level: 2nd-order around constants; scale L and Lt."""
    c  = P.c                     # [B,D]
    L  = P.L                     # [B,D,V] (0: t, 1..: x)
    Lt = P.Lt                    # [B,D,V] (0: t^2, 1..: t x_j)
    
    # ---- pointwise derivatives at c ----
    s0 = f(c)                    # [B,D]
    s1 = df(c)                   # [B,D]
    s2 = fpp(c)                  # [B,D]

    # ---- Kept polynomial ----
    a = L[:, :, 0]               # [B,D]      (L_t)
    b = L[:, :, 1:]              # [B,D,V-1]  (L_x)

    # c_out
    c_out = s0

    # L_out = f'(c) * L
    L_out = s1[:, :, None] * L

    # Lt_out = f'(c) * Lt + 0.5 f''(c) * [a^2 t^2 + 2 a t (b·x)]
    Lt_out = s1[:, :, None] * Lt
    Lt_out = Lt_out.at[:, :, 0].add(0.5 * s2 * (a * a))                    # t^2 coeff
    Lt_out = Lt_out.at[:, :, 1:].add((s2[:, :, None] * (a[:, :, None] * b)))  # t x_j coeffs

    P_keep = QuadPoly(c_out, L_out, Lt_out, P.out_shape)
    return P_keep


# ---------- PyTree registrations ----------
def _quadpoly_flatten(p: QuadPoly):
    return ((p.c, p.L, p.Lt), p.out_shape)
def _quadpoly_unflatten(aux, children):
    c, L, Lt = children
    return QuadPoly(c, L, Lt, aux)

jax.tree_util.register_pytree_node(QuadPoly, _quadpoly_flatten, _quadpoly_unflatten)
