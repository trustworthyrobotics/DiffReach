from __future__ import annotations
from dataclasses import dataclass
from math import prod

import jax
import jax.numpy as jnp

Array = jnp.ndarray
M = 1e5


# =========================
# Interval with state axis
# =========================
@dataclass
class Interval:
    """
    State-parallel interval.
    Shapes: lo, hi: [B, D]
    """
    lo: Array
    hi: Array
    out_shape: tuple[int, ...] | None = None
    # def __post_init__(self):
    #     # Sanity check, no inf or NaN
    #     assert jnp.all(jnp.isfinite(self.lo)) and jnp.all(jnp.isfinite(self.hi))

    def __post_init__(self):
        if self.out_shape is not None:
            shape = tuple(int(s) for s in self.out_shape)
            if prod(shape) != self.lo.shape[1]:
                raise ValueError(
                    f"Interval out_shape={shape} incompatible with flat dim {self.lo.shape[1]}"
                )
            self.out_shape = None if len(shape) <= 1 else shape

    # ---- Constructors ----
    @staticmethod
    def zero(B: int, D: int, dtype=jnp.float32) -> "Interval":
        z = jnp.zeros((B, D), dtype)
        return Interval(z, z)

    @staticmethod
    def zeros(shape, dtype=jnp.float32) -> "Interval":
        z = jnp.zeros(shape, dtype)
        return Interval(z, z)

    @staticmethod
    def zeros_like(ref: Array) -> "Interval":
        z = jnp.zeros_like(ref)
        return Interval(z, z)

    @staticmethod
    def unit(B: int, D: int, dtype=jnp.float32) -> "Interval":
        o = jnp.ones((B, D), dtype)
        return Interval(-o, o)

    @staticmethod
    def unit_like(ref: Array) -> "Interval":
        o = jnp.ones_like(ref)
        return Interval(-o, o)

    @staticmethod
    def from_scalar_bounds(lo: Array, hi: Array, D: int) -> "Interval":
        """Expand per-batch scalars [B] to [B,D]."""
        assert lo.ndim == hi.ndim == 1 and lo.shape == hi.shape
        B = lo.shape[0]
        lo_bd = jnp.broadcast_to(lo[:, None], (B, D))
        hi_bd = jnp.broadcast_to(hi[:, None], (B, D))
        return Interval(lo_bd, hi_bd)

    @staticmethod
    def unit_t(B: int, D: int, h: float, dtype=jnp.float32) -> "Interval":
        """Time interval [0,h], broadcast to [B,D]."""
        lo = jnp.zeros((B,), dtype)
        hi = jnp.full((B,), h, dtype)
        return Interval.from_scalar_bounds(lo, hi, D)

    def midpoint(self) -> Array:
        return 0.5 * (self.lo + self.hi)

    def radius(self) -> Array:
        return 0.5 * (self.hi - self.lo)

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return self.out_shape if self.out_shape is not None else (self._flat_dim(self.lo),)

    @staticmethod
    def _flat_dim(arr: Array) -> int:
        return int(arr.shape[1]) if getattr(arr, "ndim", 0) >= 2 else 1

    def clone(self) -> "Interval":
        return Interval(jnp.array(self.lo, copy=True), jnp.array(self.hi, copy=True), self.out_shape)

    def with_shape(self, out_shape: tuple[int, ...] | None) -> "Interval":
        return Interval(self.lo, self.hi, out_shape)

    @staticmethod
    def _normalize_shape(shape: tuple[int, ...] | None, flat_dim: int) -> tuple[int, ...] | None:
        if shape is None:
            return None
        shape = tuple(int(s) for s in shape)
        if prod(shape) != flat_dim:
            raise ValueError(f"Shape {shape} is incompatible with flat dim {flat_dim}")
        return None if len(shape) <= 1 else shape

    def _reshape_lohi(self) -> tuple[Array, Array]:
        shape = self.logical_shape
        return (
            self.lo.reshape((self.lo.shape[0], *shape)),
            self.hi.reshape((self.hi.shape[0], *shape)),
        )

    @staticmethod
    def _from_shaped(lo: Array, hi: Array, out_shape: tuple[int, ...] | None) -> "Interval":
        flat = int(prod(lo.shape[1:])) if lo.ndim > 1 else 1
        lo_flat = lo.reshape((lo.shape[0], flat))
        hi_flat = hi.reshape((hi.shape[0], flat))
        return Interval(lo_flat, hi_flat, Interval._normalize_shape(out_shape, flat))

    def reshape(self, new_shape: tuple[int, ...]) -> "Interval":
        new_shape = tuple(int(s) for s in new_shape)
        if prod(new_shape) != self.lo.shape[1]:
            raise ValueError(f"Cannot reshape interval of flat dim {self.lo.shape[1]} to {new_shape}")
        return Interval(self.lo, self.hi, Interval._normalize_shape(new_shape, self.lo.shape[1]))

    def squeeze(self, axes: tuple[int, ...] | None = None) -> "Interval":
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
        return Interval(self.lo, self.hi, Interval._normalize_shape(kept, self.lo.shape[1]))

    def broadcast_in_dim(self, shape: tuple[int, ...], broadcast_dimensions: tuple[int, ...]) -> "Interval":
        shape = tuple(int(s) for s in shape)
        in_shape = self.logical_shape
        if len(broadcast_dimensions) == 0 and self._flat_dim(self.lo) == 1:
            scalar_shape = (self.lo.shape[0],) + (1,) * len(shape)
            lo_exp = self.lo[:, :1].reshape(scalar_shape)
            hi_exp = self.hi[:, :1].reshape(scalar_shape)
            lo_out = jnp.broadcast_to(lo_exp, (self.lo.shape[0], *shape))
            hi_out = jnp.broadcast_to(hi_exp, (self.hi.shape[0], *shape))
            return Interval._from_shaped(lo_out, hi_out, shape)
        if len(in_shape) != len(broadcast_dimensions):
            raise ValueError(
                f"broadcast_dimensions={broadcast_dimensions} incompatible with input shape {in_shape}"
            )
        expanded = [1] * len(shape)
        for in_axis, out_axis in enumerate(broadcast_dimensions):
            expanded[int(out_axis)] = in_shape[in_axis]
        lo_view, hi_view = self._reshape_lohi()
        lo_exp = lo_view.reshape((self.lo.shape[0], *expanded))
        hi_exp = hi_view.reshape((self.hi.shape[0], *expanded))
        lo_out = jnp.broadcast_to(lo_exp, (self.lo.shape[0], *shape))
        hi_out = jnp.broadcast_to(hi_exp, (self.hi.shape[0], *shape))
        return Interval._from_shaped(lo_out, hi_out, shape)

    def slice(
        self,
        starts: int | tuple[int, ...],
        limits: int | tuple[int, ...],
        strides: tuple[int, ...] | None = None,
    ) -> "Interval":
        if not isinstance(starts, tuple):
            return Interval(self.lo[:, starts:limits], self.hi[:, starts:limits], self.out_shape)
        shape = self.logical_shape
        rank = len(shape)
        if strides is None:
            strides = (1,) * rank
        elif len(strides) == 1 and rank > 1:
            strides = tuple(int(strides[0]) for _ in range(rank))
        if not (len(starts) == len(limits) == len(strides) == rank):
            raise ValueError("Slice parameters must have the same rank")
        lo_view, hi_view = self._reshape_lohi()
        slicers = tuple(slice(int(s), int(l), int(st)) for s, l, st in zip(starts, limits, strides))
        lo_out = lo_view[(slice(None),) + slicers]
        hi_out = hi_view[(slice(None),) + slicers]
        return Interval._from_shaped(lo_out, hi_out, lo_out.shape[1:])

    @staticmethod
    def concat(xs: list["Interval"], axis: int = 0) -> "Interval":
        if not xs:
            raise ValueError("Interval.concat requires at least one input")
        shapes = [x.logical_shape for x in xs]
        rank = len(shapes[0])
        if any(len(shape) != rank for shape in shapes):
            raise ValueError(f"Cannot concatenate intervals of different ranks: {shapes}")
        lo_views = [x.lo.reshape((x.lo.shape[0], *shape)) for x, shape in zip(xs, shapes)]
        hi_views = [x.hi.reshape((x.hi.shape[0], *shape)) for x, shape in zip(xs, shapes)]
        lo_out = jnp.concatenate(lo_views, axis=axis + 1)
        hi_out = jnp.concatenate(hi_views, axis=axis + 1)
        return Interval._from_shaped(lo_out, hi_out, lo_out.shape[1:])

    def append_zero_var(self, n: int) -> "Interval":
        """Append a zero coefficient variable (increase D by n)."""
        B, D = self.lo.shape
        lo_new = jnp.concatenate([self.lo, jnp.zeros((B, n), dtype=self.lo.dtype)], axis=1)
        hi_new = jnp.concatenate([self.hi, jnp.zeros((B, n), dtype=self.hi.dtype)], axis=1)
        return Interval(lo_new, hi_new)

    # ---- Arithmetic ----
    def add(self, other: "Interval|Array|float") -> "Interval":
        if isinstance(other, Interval):
            out_shape = self.out_shape if self._flat_dim(self.lo) >= self._flat_dim(other.lo) else other.out_shape
            return Interval(self.lo + other.lo, self.hi + other.hi, out_shape)
        return Interval(self.lo + other, self.hi + other, self.out_shape)

    def sub(self, other: "Interval|Array|float") -> "Interval":
        if isinstance(other, Interval):
            out_shape = self.out_shape if self._flat_dim(self.lo) >= self._flat_dim(other.lo) else other.out_shape
            return Interval(self.lo - other.hi, self.hi - other.lo, out_shape)
        return Interval(self.lo - other, self.hi - other, self.out_shape)

    def scale(self, k: "float|Array") -> "Interval":
        lo = self.lo * k
        hi = self.hi * k
        return Interval(jnp.minimum(lo, hi), jnp.maximum(lo, hi), self.out_shape)

    def mul(self, other: "Interval|Array|float") -> "Interval":
        if isinstance(other, Interval):
            a = jnp.stack(
                [self.lo * other.lo, self.lo * other.hi,
                 self.hi * other.lo, self.hi * other.hi],
                axis=0
            )
            out_shape = self.out_shape if self._flat_dim(self.lo) >= self._flat_dim(other.lo) else other.out_shape
            return Interval(jnp.min(a, axis=0), jnp.max(a, axis=0), out_shape)
        else:
            return self.scale(other)

    def affine(self, A: Array) -> "Interval":
        """
        Affine map: self ↦ A·self
        A: [B,D,D], b: [B,D]
        """
        pos_mask = A >= 0 # [B,D,D]
        neg_mask = 1 - pos_mask
        R_other_lo = self.lo[..., None]                      # [B,D,1]
        R_other_hi = self.hi[..., None]                      # [B,D,1]
        pos_A = A * pos_mask
        neg_A = A * neg_mask
        R_extra_lo = jnp.sum(pos_A @ R_other_lo + neg_A @ R_other_hi, axis=-1)   # [B,D]
        R_extra_hi = jnp.sum(pos_A @ R_other_hi + neg_A @ R_other_lo, axis=-1)   # [B,D]
        
        return Interval(R_extra_lo, R_extra_hi, self.out_shape)

    def pow(self, n: int):
        """
        Elementwise power for integer n ≥ 0.
        Handles even/odd n and sign changes like Flow*.
        """
        if n == 0:
            # x^0 = 1 for any nonempty interval
            ones = jnp.ones_like(self.lo)
            return Interval(ones, ones, self.out_shape)

        if n == 1:
            return Interval(self.lo, self.hi, self.out_shape)

        lo, hi = self.lo, self.hi

        # even power -> nonnegative, possible zero crossing
        if n % 2 == 0:
            lo_pow = jnp.minimum(lo ** n, hi ** n)
            hi_pow = jnp.maximum(lo ** n, hi ** n)
            # if interval crosses zero, lower bound is 0
            lo_pow = jnp.where((lo <= 0) & (hi >= 0), 0.0, lo_pow)
        else:
            # odd power -> monotonic
            lo_pow = lo ** n
            hi_pow = hi ** n

        return Interval(lo_pow, hi_pow, self.out_shape)

    def recip(self) -> "Interval":
        """
        Interval reciprocal 1 / self, matching Flow* semantics:
        if interval crosses 0, result = unbounded (-M, +M).
        """
        lo_d, hi_d = self.lo, self.hi

        # check zero crossing
        crosses_zero = (lo_d <= 0) & (hi_d >= 0)

        # safe reciprocal bounds
        inv_lo = 1.0 / hi_d
        inv_hi = 1.0 / lo_d
        inv_lo, inv_hi = jnp.minimum(inv_lo, inv_hi), jnp.maximum(inv_lo, inv_hi)
        result = Interval(inv_lo, inv_hi, self.out_shape)

        # replace with unbounded interval if crosses zero
        lo_res = jnp.where(crosses_zero, -M, result.lo)
        hi_res = jnp.where(crosses_zero, M, result.hi)
        return Interval(lo_res, hi_res, self.out_shape)

    def div(self, other):
        """
        Interval division self / other, matching Flow* semantics:
        if denominator crosses 0, result = unbounded (-M, +M).
        """
        lo_d, hi_d = other.lo, other.hi

        # check zero crossing
        crosses_zero = (lo_d <= 0) & (hi_d >= 0)

        # safe reciprocal bounds
        inv_lo = 1.0 / hi_d
        inv_hi = 1.0 / lo_d
        inv_lo, inv_hi = jnp.minimum(inv_lo, inv_hi), jnp.maximum(inv_lo, inv_hi)
        inv = Interval(inv_lo, inv_hi, other.out_shape)

        result = self.mul(inv)

        # replace with unbounded interval if denominator crosses zero
        lo_res = jnp.where(crosses_zero, -M, result.lo)
        hi_res = jnp.where(crosses_zero, M, result.hi)
        return Interval(lo_res, hi_res, self.out_shape)

    def square(self) -> "Interval":
        l2 = self.lo * self.lo
        u2 = self.hi * self.hi
        pos = (self.lo >= 0)
        neg = (self.hi <= 0)
        lo = jnp.where(pos, l2, jnp.where(neg, u2, jnp.zeros_like(l2)))
        hi = jnp.where(pos, u2, jnp.where(neg, l2, jnp.maximum(l2, u2)))
        return Interval(lo, hi, self.out_shape)

    def enlarge(self, eps: float|Array) -> "Interval":
        """Enlarge interval by eps in all directions."""
        return Interval(self.lo - eps, self.hi + eps, self.out_shape)

    # utilities
    def width(self) -> Array:
        return self.hi - self.lo

    def subseteq(self, other: "Interval") -> bool:
        """Check if self ⊆ other elementwise over batch."""
        return jnp.all(self.lo >= other.lo, axis=-1) & jnp.all(self.hi <= other.hi, axis=-1)

    def subseteq_elem(self, other: "Interval") -> Array:
        """Elementwise check if self ⊆ other. Returns [B, D] boolean array."""
        return (self.lo >= other.lo) & (self.hi <= other.hi)

    def to_fp64(self) -> "Interval":
        return Interval(self.lo.astype(jnp.float64), self.hi.astype(jnp.float64), self.out_shape)

    def is_zero(self) -> Array:
        """Check if interval is identically zero over all batches and dimensions."""
        return jnp.allclose(self.lo, 0) & jnp.allclose(self.hi, 0)

    def where(self, mask: Array, other: "Interval") -> "Interval":
        lo = jnp.where(mask, self.lo, other.lo)
        hi = jnp.where(mask, self.hi, other.hi)
        return Interval(lo, hi, self.out_shape)
    
    def unsqueeze(self, axis: int) -> "Interval":
        return Interval(jnp.expand_dims(self.lo, axis), jnp.expand_dims(self.hi, axis), self.out_shape)

    def cumsum(self, axis: int | None = None) -> "Interval":
        """
        Prefix sum along the state dimension.
        Currently only supports 1D input semantics, i.e. axis=None or axis=1.
        """
        if axis not in (None, 1):
            raise NotImplementedError("Interval.cumsum currently only supports axis=None for 1D inputs")
        return Interval(jnp.cumsum(self.lo, axis=1), jnp.cumsum(self.hi, axis=1), self.out_shape)

    def reduce_sum(self, axes: tuple[int, ...]) -> "Interval":
        shape = self.logical_shape
        rank = len(shape)
        norm_axes = tuple(ax if ax >= 0 else rank + ax for ax in axes)
        lo_view, hi_view = self._reshape_lohi()
        lo_out = jnp.sum(lo_view, axis=tuple(ax + 1 for ax in norm_axes))
        hi_out = jnp.sum(hi_view, axis=tuple(ax + 1 for ax in norm_axes))
        out_shape = tuple(dim for idx, dim in enumerate(shape) if idx not in norm_axes)
        return Interval._from_shaped(lo_out, hi_out, out_shape or (1,))

    def log(self, prefix: str = "Interval", dim=None):
        if dim is None:
            jax.debug.print(f"{prefix}: lo={self.lo.tolist()}, hi={self.hi.tolist()}")
        else:
            jax.debug.print(f"{prefix}: lo={self.lo[:, dim].tolist()}, hi={self.hi[:, dim].tolist()}")
        return

# ======================
# Interval utilities for non-polynomial ops
# ======================
def _contains_multiple(a, b, offset, period, eps=1e-6):
    """
    True iff [a,b] contains some {offset + k*period}, elementwise for arrays.
    eps handles float32 rounding near boundaries.
    """
    lo = jnp.minimum(a, b)
    hi = jnp.maximum(a, b)

    # If interval covers (almost) a whole period, it's certainly true
    covers_period = (hi - lo) >= (period - eps)

    kmin = jnp.ceil((lo - offset - eps) / period)
    kmax = jnp.floor((hi - offset + eps) / period)
    return covers_period | (kmin <= kmax)

def iv_neg(I: Interval) -> Interval:
    return Interval(-I.hi, -I.lo)

def iv_sin(I: Interval) -> Interval:
    a, b = I.lo, I.hi
    sa, sb = jnp.sin(a), jnp.sin(b)
    has_pos1 = _contains_multiple(a, b, jnp.pi/2, 2*jnp.pi)
    has_neg1 = _contains_multiple(a, b, 3*jnp.pi/2, 2*jnp.pi)
    lo = jnp.minimum(jnp.minimum(sa, sb), jnp.where(has_neg1, -jnp.ones_like(sa), jnp.inf))
    hi = jnp.maximum(jnp.maximum(sa, sb), jnp.where(has_pos1,  jnp.ones_like(sa), -jnp.inf))
    lo = jnp.where(jnp.isinf(lo), jnp.minimum(sa, sb), lo)
    hi = jnp.where(jnp.isinf(hi), jnp.maximum(sa, sb), hi)
    return Interval(lo, hi)

def iv_cos(I: Interval) -> Interval:
    a, b = I.lo, I.hi
    ca, cb = jnp.cos(a), jnp.cos(b)
    has_pos1 = _contains_multiple(a, b, 0.0, 2*jnp.pi)
    has_neg1 = _contains_multiple(a, b, jnp.pi, 2*jnp.pi)
    lo = jnp.minimum(jnp.minimum(ca, cb), jnp.where(has_neg1, -jnp.ones_like(ca), jnp.inf))
    hi = jnp.maximum(jnp.maximum(ca, cb), jnp.where(has_pos1,  jnp.ones_like(ca), -jnp.inf))
    lo = jnp.where(jnp.isinf(lo), jnp.minimum(ca, cb), lo)
    hi = jnp.where(jnp.isinf(hi), jnp.maximum(ca, cb), hi)
    return Interval(lo, hi)

def iv_tanh(I: Interval) -> Interval:
    return Interval(jnp.tanh(I.lo), jnp.tanh(I.hi))

def iv_tan(Iv: Interval) -> Interval:
    """
    Image of tan over interval Iv.
    Preconditions: Iv must not cross a pole x = pi/2 + k*pi.
    On each pole-free branch tan is strictly increasing, so min/max are at the endpoints.
    """
    halfpi = jnp.pi / 2.0
    pi = jnp.pi

    # Detect any pole (pi/2 + k*pi) in [lo, hi]
    k_lo = jnp.ceil((Iv.lo - halfpi) / pi)
    k_hi = jnp.floor((Iv.hi - halfpi) / pi)
    crosses_pole = k_hi >= k_lo

    def _raise_if_pole(flag):
        if bool(jnp.any(flag)):
            raise ValueError("tan: interval crosses a pole (pi/2 + k*pi).")
    if hasattr(jax, "debug") and hasattr(jax.debug, "callback"):
        jax.debug.callback(_raise_if_pole, crosses_pole)

    lo_img = jnp.tan(Iv.lo)
    hi_img = jnp.tan(Iv.hi)
    lo_out = jnp.minimum(lo_img, hi_img)
    hi_out = jnp.maximum(lo_img, hi_img)
    return Interval(lo_out, hi_out)

def iv_exp(I: Interval) -> Interval:
    return Interval(jnp.exp(I.lo), jnp.exp(I.hi))

def iv_log(I: Interval) -> Interval:
    eps = jnp.finfo(I.lo.dtype).tiny
    lo = jnp.maximum(I.lo, eps)
    hi = jnp.maximum(I.hi, lo)
    return Interval(jnp.log(lo), jnp.log(hi))

def iv_sqrt(I: Interval) -> Interval:
    lo = jnp.maximum(I.lo, 0.0)
    hi = jnp.maximum(I.hi, lo)
    return Interval(jnp.sqrt(lo), jnp.sqrt(hi))

def iv_recip(I: Interval) -> Interval:
    crosses_zero = (I.lo <= 0.0) & (I.hi >= 0.0)
    lo = 1.0 / I.hi
    hi = 1.0 / I.lo
    lo2 = jnp.minimum(lo, hi)
    hi2 = jnp.maximum(lo, hi)
    lo_out = jnp.where(crosses_zero, -jnp.inf, lo2)
    hi_out = jnp.where(crosses_zero,  jnp.inf, hi2)
    return Interval(lo_out, hi_out)

def iv_pow_scalar(I: Interval, p: float) -> Interval:
    if float(p).is_integer():
        k = int(p)
        if k == 0:
            one = jnp.ones_like(I.lo)
            return Interval(one, one)
        if k < 0:
            pos = iv_pow_scalar(I, -k)
            return iv_recip(pos)
        if k == 1:
            return Interval(I.lo, I.hi)
        if k == 2:
            return I.square()
        out = Interval(I.lo, I.hi)
        for _ in range(k - 1):
            out = out.mul(I)
        return out
    else:
        lo = jnp.maximum(I.lo, 0.0)
        hi = jnp.maximum(I.hi, lo)
        return Interval(lo**p, hi**p)

# f'' and f^{(3)} interval helpers for common unaries
def iv_sin_fpp(I: Interval) -> Interval:  # -sin
    return iv_neg(iv_sin(I))
def iv_sin_fppp(I: Interval) -> Interval:  # -cos
    return iv_neg(iv_cos(I))

def iv_cos_fpp(I: Interval) -> Interval:  # -cos
    return iv_neg(iv_cos(I))
def iv_cos_fppp(I: Interval) -> Interval:  # sin
    return iv_sin(I)

def iv_exp_fpp(I: Interval) -> Interval:  # exp
    return iv_exp(I)
def iv_exp_fppp(I: Interval) -> Interval:  # exp
    return iv_exp(I)

def iv_tanh_fpp(I: Interval) -> Interval:  # -2 tanh (1 - tanh^2)
    T = iv_tanh(I); T2 = T.square(); one_minus_T2 = Interval(1.0 - T2.hi, 1.0 - T2.lo)
    return T.mul(one_minus_T2).mul(Interval(jnp.asarray(-2.0, T.lo.dtype), jnp.asarray(-2.0, T.hi.dtype)))
def iv_tanh_fppp(I: Interval) -> Interval:  # 2(1 - 4t^2 + 3 t^4)
    T = iv_tanh(I); T2 = T.square(); T4 = T2.square()
    one = Interval(jnp.ones_like(T.lo), jnp.ones_like(T.hi))
    fourT2 = T2.mul(Interval(jnp.asarray(4.0, T.lo.dtype), jnp.asarray(4.0, T.hi.dtype)))
    threeT4 = T4.mul(Interval(jnp.asarray(3.0, T.lo.dtype), jnp.asarray(3.0, T.hi.dtype)))
    inner = one.add(iv_neg(fourT2)).add(threeT4)
    two = Interval(jnp.asarray(2.0, T.lo.dtype), jnp.asarray(2.0, T.hi.dtype))
    return inner.mul(two)

def iv_log_fpp(I: Interval) -> Interval:  # -1/x^2
    inv2 = iv_pow_scalar(I, -2.0)
    return iv_neg(inv2)
def iv_log_fppp(I: Interval) -> Interval:  # 2/x^3
    return iv_pow_scalar(I, -3.0).mul(Interval(jnp.asarray(2.0, I.lo.dtype), jnp.asarray(2.0, I.hi.dtype)))

def iv_sqrt_fpp(I: Interval) -> Interval:  # -1/(4 x^{3/2})
    pow_m32 = iv_pow_scalar(I, -1.5)
    return pow_m32.mul(Interval(jnp.asarray(-0.25, I.lo.dtype), jnp.asarray(-0.25, I.hi.dtype)))
def iv_sqrt_fppp(I: Interval) -> Interval:  # 3/(8 x^{5/2})
    pow_m52 = iv_pow_scalar(I, -2.5)
    return pow_m52.mul(Interval(jnp.asarray(3.0/8.0, I.lo.dtype), jnp.asarray(3.0/8.0, I.hi.dtype)))

def iv_tan_fpp(I: Interval) -> Interval:  # 2 tan sec^2
    Tan = iv_tan(I); Cos = iv_cos(I); Sec = iv_recip(Cos); Sec2 = Sec.square()
    two = Interval(jnp.asarray(2.0, I.lo.dtype), jnp.asarray(2.0, I.hi.dtype))
    return two.mul(Tan.mul(Sec2))
def iv_tan_fppp(I: Interval) -> Interval:  # 2 sec^2 (1 + 2 tan^2)
    Tan = iv_tan(I); Tan2 = Tan.square(); Cos = iv_cos(I); Sec = iv_recip(Cos); Sec2 = Sec.square()
    one = Interval(jnp.ones_like(I.lo), jnp.ones_like(I.hi))
    inner = one.add(Tan2.mul(Interval(jnp.asarray(2.0, I.lo.dtype), jnp.asarray(2.0, I.hi.dtype))))
    two = Interval(jnp.asarray(2.0, I.lo.dtype), jnp.asarray(2.0, I.hi.dtype))
    return two.mul(Sec2.mul(inner))

def iv_intersect(I1: Interval, I2: Interval) -> Interval:
    # Intersection of two axis-aligned intervals (elementwise).
    lo = jnp.maximum(I1.lo, I2.lo)
    hi = jnp.minimum(I1.hi, I2.hi)
    return Interval(lo, hi)

def iv_sub_nonneg(A: Interval, B: Interval) -> Interval:
    """
    Nonnegative interval "difference": over-approx of {a - b | a∈A, b∈B, and ≥ 0 }.
    Used to remove a known-kept subset from a larger magnitude bound (clamped at 0).
    """
    lo = jnp.maximum(0.0, A.lo - B.hi)
    hi = jnp.maximum(0.0, A.hi - B.lo)
    return Interval(lo, hi)


# ---------- PyTree registrations ----------
def _interval_flatten(i: Interval):
    return ((i.lo, i.hi), i.out_shape)
def _interval_unflatten(aux, children):
    lo, hi = children
    return Interval(lo, hi, aux)
jax.tree_util.register_pytree_node(Interval, _interval_flatten, _interval_unflatten)
