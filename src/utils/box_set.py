from typing import Any, Callable, Dict, List, Tuple
import jax
import jax.numpy as jnp
from functools import partial

def split_initial_box(x0_lo: jnp.ndarray,
                      x0_hi: jnp.ndarray,
                      splits_per_dim) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Split an initial hyper-rectangle into a uniform grid of sub-boxes.

    Args:
      x0_lo: (B, D) lower bounds (float32).
      x0_hi: (B, D) upper bounds (float32).
      splits_per_dim: int or sequence[int] of length D (number of splits per dim).

    Returns:
      parts_lo, parts_hi: both (B*M, D) where M = prod(splits_per_dim).
    """
    assert x0_lo.ndim == 2 and x0_hi.ndim == 2 and x0_lo.shape == x0_hi.shape
    B, D = x0_lo.shape
    # Canonicalize splits to a static tuple so we can jit efficiently.
    if isinstance(splits_per_dim, int):
        splits = tuple([int(splits_per_dim)] * D)
    else:
        splits = tuple(int(s) for s in splits_per_dim)
        assert len(splits) == D, "splits_per_dim must have length D"

    return _split_initial_box_jit(x0_lo, x0_hi, splits)

@partial(jax.jit, static_argnames=("splits",))
def _split_initial_box_jit(x0_lo: jnp.ndarray,
                           x0_hi: jnp.ndarray,
                           splits: tuple[int, ...]) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Build grid of integer indices of shape (M, D) using jnp.indices (static shape).
    idx_grid = jnp.indices(splits, dtype=x0_lo.dtype)        # (D, s1, s2, ..., sD)
    idx = jnp.reshape(jnp.moveaxis(idx_grid, 0, -1), (-1, len(splits)))  # (M, D)
    s = jnp.asarray(splits, x0_lo.dtype)[None, :]            # (1, D)

    # Fractions for each sub-box along each dim
    frac_lo = idx / s                                        # (M, D)
    frac_hi = (idx + 1.0) / s                                # (M, D)

    # Broadcast over batch to get all sub-box lows/highs
    widths = (x0_hi - x0_lo)                                 # (B, D)
    parts_lo = x0_lo[:, None, :] + widths[:, None, :] * frac_lo[None, :, :]  # (B, M, D)
    parts_hi = x0_lo[:, None, :] + widths[:, None, :] * frac_hi[None, :, :]  # (B, M, D)

    # Flatten batch and grid into one list of boxes
    B = x0_lo.shape[0]
    D = x0_lo.shape[1]
    return parts_lo.reshape(-1, D), \
           parts_hi.reshape(-1, D)

# ----------------------------
# Initial set handling (with per-dim splits)
# ----------------------------
def _splits_dict_to_list(splits_dict: Dict[int,int], D_total: int) -> List[int]:
    """
    Convert a sparse dict like {0:8,1:8,2:8,3:2} into a dense list of length D_total.
    Unspecified dims default to 1. Non-positive values raise.
    """
    out = [1]*D_total
    for k, v in (splits_dict or {}).items():
        idx = int(k)
        if 0 <= idx < D_total:
            vv = int(v)
            if vv <= 0:
                raise ValueError(f"splits[{idx}] must be positive; got {vv}.")
            out[idx] = vv
    return out

def prepare_initial_sets(initial_set: jnp.ndarray,
                         splits_cfg: Dict[int,int],
                         D_total: int, dtype=jnp.float32) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    initial_set: list of [lo,hi] for D_total variables in config order.
    splits_cfg: dict[int,int] -> per-dim split counts (unspecified -> 1).

    Returns:
      parts_lo, parts_hi: jnp arrays [B, D_total] after splitting (B=∏ splits)
    """
    x0_lo = jnp.asarray(initial_set[:, 0][None, :], dtype=dtype)  # [1, D_total]
    x0_hi = jnp.asarray(initial_set[:, 1][None, :], dtype=dtype)  # [1, D_total]

    splits_per_dim = _splits_dict_to_list(splits_cfg, D_total)  # len=D_total
    parts_lo, parts_hi = split_initial_box(x0_lo, x0_hi, splits_per_dim)  # [B, D_total] each
    return parts_lo, parts_hi


def prepare_initial_set_v2(x0_lo: jnp.ndarray,
                           x0_hi: jnp.ndarray,
                           splits_cfg: Dict[int,int]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    x0_lo: (B, D) lower bounds (float32).
    x0_hi: (B, D) upper bounds (float32).
    splits_cfg: dict[int,int] -> per-dim split counts (unspecified -> 1).
    Returns:
      parts_lo, parts_hi: jnp arrays [B*M, D] after splitting (M=∏ splits)
    """
    D = x0_lo.shape[1]
    splits_per_dim = _splits_dict_to_list(splits_cfg, D)  # len=D
    parts_lo, parts_hi = split_initial_box(x0_lo, x0_hi, splits_per_dim)  # [B*M, D] each
    return parts_lo, parts_hi

# @jax.jit
def calculate_volume(lowers: jnp.ndarray, uppers: jnp.ndarray, union_init: bool = False, mode='prod', keep_time: bool = False, keep_batch: bool = False) -> jnp.ndarray:
    """Compute normalized volume across partitions and steps.
    
    Args:
        lowers: [n_partitions, n_steps, state_dim]
        uppers: [n_partitions, n_steps, state_dim]
        
    Returns:
        Scalar: The average (over steps) of the total volume (summed over partitions).
    """
    # 1. Calculate raw widths, clipping negative values to 0
    raw_widths = jnp.maximum(uppers - lowers, 0.0)  # Shape: [P, S, D]

    # 2. Calculate normalization factor based on the initial step (time index 0)
    if union_init:
        #    We look at the union of all partitions at step 0 to define the scale.
        #    lowers[:, 0, :] has shape [P, D]
        init_min = jnp.min(lowers[:, 0, :], axis=0) 
        init_max = jnp.max(uppers[:, 0, :], axis=0)
        # replace zeros with ones
        init_range = init_max - init_min
        #    Broadcast init_range: [D] -> [1, 1, D] to match [P, S, D]
        init_range = init_range[None, None, :]
    else:
        #    We look at each partition's initial box to define the scale.
        init_range = uppers[:, 0, :] - lowers[:, 0, :]  # Shape: [P, D]
        #    Broadcast init_range: [P, D] -> [P, 1, D] to match [P, S, D]
        init_range = init_range[:, None, :]
    init_range = jnp.where(init_range > 0.0, init_range, 1.0)

    # 3. Normalize widths
    #    Broadcast init_range: [D] -> [1, 1, D] to match [P, S, D]
    norm_widths = raw_widths / init_range
    norm_widths = jnp.where(norm_widths > 0.0, norm_widths, 1.0)

    # 4. Compute Volume (product of dimensions)
    #    Collapse last axis (state_dim). Result shape: [P, S]
    if mode == 'prod':
        box_volumes = jnp.prod(norm_widths, axis=-1)
    elif mode == 'sum':
        box_volumes = jnp.sum(norm_widths, axis=-1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if keep_batch:
        total_vol_per_step = box_volumes  # Shape: [P, S]
    else:
        # 5. Aggregate
        #    Sum volumes across partitions (axis 0) -> [S]
        total_vol_per_step = jnp.sum(box_volumes, axis=0)
    
    if keep_time:
        return total_vol_per_step  # Shape: [S] or [P, S]
    
    #    Average across time steps -> Scalar
    return jnp.mean(total_vol_per_step, axis=-1) # Shape: [] or [P]
