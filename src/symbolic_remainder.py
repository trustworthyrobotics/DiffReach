# Copyright (c) 2025
# Flow*-style Symbolic Remainder with Maximum-Size Window (no inverses)
# ---------------------------------------------------------------------
# This implementation mirrors the FULL list semantics when the window is not
# full, and adds Flow*'s queue-cap behavior when the window reaches size M:
# after committing a step, if the window size becomes M, we CLEAR both queues
# so the NEXT step is computed non-symbolically (no symbolic carry).
# No matrix inverses or solves are used.
#
# Public API matches the original modules:
#   - SymbolicRemainderState
#   - init_symbolic_state(batch, dim, M, *, dtype=jnp.float32)
#   - symbolic_step_linear(tmv: QuadTM, x0_tm: QuadTM,
#                          state: SymbolicRemainderState, eps=1e-12)
#        -> (S: (B,D), next_tmv: QuadTM, next_state)
#
# Per-step math (identical to full when not full):
#   A      = a_x @ L
#   Φ_new  = a_x * invS_prev   (column-normalized per row)
#   J_new  = x0_tm.R
#   Phi_upd      = [Φ_new @ P  for P in Phi_list]
#   Phi_contrib  = concat(Phi_upd[1:], Φ_new)
#   Past         = Σ_k Phi_contrib[k] @ J_list[k] (interval affine map + sum)
#   R            = Past + J_new
#   S            = ||A||_{row-1} + R
#   next_tmv.P.L = A
#   next_tmv.R   = R scaled by 1/S
#
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from src.taylor_model import QuadPoly, QuadTM, Interval

Array = jnp.ndarray


@register_pytree_node_class
@dataclass
class SymbolicRemainderState:
    """Windowed SR state with Flow* list semantics.

    - Phi_buf: (B,M,D,D)   last up to M raw Φ entries (column-normalized)
    - J_buf:   (B,M,D)     last up to M local remainders J_k
    - count:   () int32    number of valid entries (≤ M)
    - M:       int         capacity (static; aux in pytree)
    - invS:    (B,D)       previous step's 1/S
    """
    Phi_buf: Array
    J_buf: Interval
    count: Array
    M: int
    invS: Array

    # pytree protocol: keep arrays in children; keep M as aux (python int)
    def tree_flatten(self):
        children = (self.Phi_buf, self.J_buf, self.count, self.invS)
        aux_data = int(self.M)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        Phi_buf, J_buf, count, invS = children
        return cls(Phi_buf=Phi_buf, J_buf=J_buf, count=count, M=int(aux_data), invS=invS)


def init_symbolic_state(batch: int, dim: int, M: int, *, dtype=jnp.float32) -> SymbolicRemainderState:
    Phi_buf = jnp.zeros((batch, M, dim, dim), dtype)
    J_buf   = Interval.zeros((batch, M, dim), dtype)
    count   = jnp.array(0, dtype=jnp.int32)
    invS    = jnp.ones((batch, dim), dtype)
    return SymbolicRemainderState(Phi_buf=Phi_buf, J_buf=J_buf, count=count, M=M, invS=invS)

def symbolic_step_linear(
    tmv: QuadTM, x0_tm: QuadTM, st: SymbolicRemainderState,
    eval_lo: Array, eval_hi: Array, eps: float = 1e-12,
) -> Tuple[Array, QuadTM, SymbolicRemainderState]:
    """One SR step with Flow* list semantics + window cap.

    When the window is not full, this matches the full implementation exactly.
    When appending this step makes the window reach M, we clear both buffers so
    the NEXT step is non-symbolic (no carry), as described in the paper.
    """
    # Blocks
    L    = tmv.P.L[:, :, 1:]                              # (B,D,D)
    a_x  = x0_tm.P.L[:, :, 1:]                            # (B,D,D)
    A    = jnp.einsum('bij,bjk->bik', a_x, L)             # (B,D,D)

    # New column-normalized linear and local remainder
    Phi_new = a_x * st.invS[:, None, :]                   # (B,D,D)
    r_x0   = x0_tm.R      # (B,D)
    r_seed = tmv.R        # (B,D)
    seed_thru = r_seed.affine(a_x)          # (B,D)

    M = st.M
    idx = jnp.arange(M, dtype=jnp.int32)                  # (M,)

    cnt = st.count                                        # () int32
    active = (idx < cnt)                                   # (M,)

    # Left-multiply stored Φ by Φ_new (for all M, then mask to first cnt)
    Phi_upd_all = jnp.einsum('bij,bmjk->bmik', Phi_new, st.Phi_buf)  # (B,M,D,D)

    # Build Phi_contrib with Flow* alignment: concat(Phi_upd[1:], Phi_new)
    # For cnt==0, Past=0. For cnt>=1, we only use the first cnt entries.
    Phi_roll = jnp.roll(Phi_upd_all, shift=-1, axis=1)              # (B,M,D,D)
    last_active = (idx == (cnt - 1))
    Phi_contrib_all = jnp.where(
        last_active[None, :, None, None],
        Phi_new[:, None, :, :],
        Phi_roll,
    )

    Past = st.J_buf.affine(Phi_contrib_all)
    Past = Interval(Past.lo.sum(axis=1), Past.hi.sum(axis=1))  # (B,D)
    J_new = r_x0.add(seed_thru).where(cnt == 0, r_x0) 

    # Total remainder and Next seed
    next_R   = Past.add(J_new)
    Bsz, Dp, V = tmv.P.c.shape[0], tmv.P.c.shape[1], tmv.P.L.shape[2]
    Pn = QuadPoly.zeros(Bsz, Dp, V, dtype=A.dtype)
    Pn.L = Pn.L.at[:, :, 1:].set(A)
    next_tmv = QuadTM(Pn, next_R)

    # scales
    range_of_x0 = next_tmv.eval_interval(eval_lo, eval_hi)  # (B,D)
    S = jnp.maximum(jnp.abs(range_of_x0.hi), jnp.abs(range_of_x0.lo))    # (B,D)
    invS = 1.0 / (S + eps)
    
    # rows scaled by invS
    next_tmv = next_tmv.scale(invS)

    # Commit: update first `cnt` slots to Phi_upd_all; set slot `cnt` to Φ_new; J_buf[cnt]=J_new
    active_mask = active[None, :, None, None]
    Phi_buf_next = jnp.where(active_mask, Phi_upd_all, st.Phi_buf)

    ins_mask_phi = (idx == cnt)[None, :, None, None]
    Phi_buf_next = jnp.where(ins_mask_phi, Phi_new[:, None, :, :], Phi_buf_next)

    J_buf_next = st.J_buf
    ins_mask_j = (idx == cnt)[None, :, None]
    J_buf_next = J_new.unsqueeze(1).where(ins_mask_j, J_buf_next)

    count_next = jnp.minimum(cnt + 1, jnp.array(M, jnp.int32))

    # If we just reached size M, clear both buffers so the NEXT step is non-symbolic
    just_full = jnp.equal(count_next, jnp.array(M, jnp.int32))

    def cleared():
        return (
            jnp.zeros_like(Phi_buf_next),
            Interval.zeros_like(J_buf_next.lo),
            jnp.array(0, jnp.int32),
        )

    def keep():
        return (Phi_buf_next, J_buf_next, count_next)

    Phi_final, J_final, count_final = jax.lax.cond(just_full, cleared, keep)

    next_state = SymbolicRemainderState(
        Phi_buf=Phi_final,
        J_buf=J_final,
        count=count_final,
        M=M,
        invS=invS,
    )

    return S, next_tmv, next_state
