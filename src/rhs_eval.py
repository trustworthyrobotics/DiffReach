from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple

import jax
import jax.numpy as jnp
import jax.lax as jlax


from src.interval import *
from src.polynomial import QuadPoly, poly_unary
from src.taylor_model import QuadTM, tm_unary_lagrange

import src.settings as settings

Array = jnp.ndarray


# ======================
# JAXPR tracing
# ======================
@dataclass
class ClosedProg:
    jaxpr: Any
    consts: Any
    D: int

def _trace_prog(f: Callable, D: int) -> ClosedProg:
    x0 = jnp.zeros((D,), jnp.float32)
    closed = jax.make_jaxpr(f)(x0)  # ClosedJaxpr
    return ClosedProg(closed.jaxpr, closed.consts, D)


# ----------------------
# Poly interpreter (quadratic version)
# ----------------------
def build_rhs_poly_from_prog(prog: ClosedProg, V: int) -> Callable[[QuadTM], QuadPoly]:
    """Evaluate RHS into kept-affine QuadPoly (Lt=0), with truncated affine products, incl. non-polys."""
    jaxpr, consts, D = prog.jaxpr, prog.consts, prog.D

    def _broadcast_poly_to_shape(x: QuadPoly, target_shape: tuple[int, ...]) -> QuadPoly:
        if x.logical_shape == target_shape:
            return x
        if x.D == 1 and target_shape != (1,):
            return x.broadcast_in_dim(target_shape, ())
        if len(x.logical_shape) > len(target_shape):
            raise ValueError(f"Cannot broadcast shape {x.logical_shape} to {target_shape}")
        dims = tuple(range(len(target_shape) - len(x.logical_shape), len(target_shape)))
        return x.broadcast_in_dim(target_shape, dims)

    def _broadcast_poly_pair(a: QuadPoly, b: QuadPoly) -> tuple[QuadPoly, QuadPoly]:
        target_shape = jnp.broadcast_shapes(a.logical_shape, b.logical_shape)
        return _broadcast_poly_to_shape(a, target_shape), _broadcast_poly_to_shape(b, target_shape)

    def _eval_poly_cumsum_arg(arg, axis, ref_poly):
        if axis not in (0, None):
            raise NotImplementedError(f"Only axis=None/0 cumsum is supported, got axis={axis}")
        if isinstance(arg, QuadPoly):
            return arg.cumsum(axis=None)
        if isinstance(arg, QuadTM):
            return arg.P.cumsum(axis=None)
        return jnp.cumsum(arg, axis=axis)

    def _poly_reshape(ins, eqn, P_in):
        del P_in
        return ins[0].reshape(tuple(eqn.params["new_sizes"]))

    def _poly_squeeze(ins, eqn, P_in):
        del P_in
        return ins[0].squeeze(tuple(eqn.params.get("dimensions", ())))

    def _poly_convert(ins, eqn, P_in):
        del eqn
        return ins[0] if isinstance(ins[0], QuadPoly) else QuadPoly.to_poly_like(ins[0], P_in)

    def _poly_broadcast(ins, eqn, P_in):
        broadcast_shape = tuple(eqn.params["shape"])
        if isinstance(ins[0], QuadPoly):
            return ins[0].broadcast_in_dim(broadcast_shape, tuple(eqn.params.get("broadcast_dimensions", ())))
        if broadcast_shape == (1,):
            return QuadPoly.to_poly_like(ins[0], P_in)
        return QuadPoly.to_poly_like(jnp.broadcast_to(ins[0], broadcast_shape), P_in)

    def _poly_slice(ins, eqn, P_in):
        del P_in
        starts  = tuple(eqn.params.get("start_indices", ()))
        limits  = tuple(eqn.params.get("limit_indices", ()))
        strides = eqn.params.get("strides", None) or (1,)
        return QuadPoly.slice(ins[0], starts if len(starts) > 1 else int(starts[0]), limits if len(limits) > 1 else int(limits[0]), strides)

    def _poly_add(ins, eqn, P_in):
        del eqn
        a, b = _broadcast_poly_pair(QuadPoly.to_poly_like(ins[0], P_in), QuadPoly.to_poly_like(ins[1], P_in))
        return a.add(b)

    def _poly_sub(ins, eqn, P_in):
        del eqn
        a, b = _broadcast_poly_pair(QuadPoly.to_poly_like(ins[0], P_in), QuadPoly.to_poly_like(ins[1], P_in))
        return a.sub(b)

    def _poly_mul(ins, eqn, P_in):
        del eqn
        a, b = _broadcast_poly_pair(QuadPoly.to_poly_like(ins[0], P_in), QuadPoly.to_poly_like(ins[1], P_in))
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            return a._mul_trunc1(b)
        return a.mul(b)

    def _poly_neg(ins, eqn, P_in):
        del eqn
        return QuadPoly.to_poly_like(ins[0], P_in).scale(-1.0)

    def _poly_unary_handler(fn, fp, fpp):
        def _handler(ins, eqn, P_in):
            del eqn
            return poly_unary(QuadPoly.to_poly_like(ins[0], P_in), fn, fp, fpp)
        return _handler

    def _poly_div(ins, eqn, P_in):
        del eqn
        a, b = ins
        if isinstance(b, QuadPoly):
            Pa = QuadPoly.to_poly_like(a, P_in)
            Pb = QuadPoly.to_poly_like(b, P_in)
            return Pa.div(Pb)
        if isinstance(a, QuadPoly):
            return QuadPoly.to_poly_like(a, P_in).scale(1.0 / b)
        raise TypeError(
            f"Unsupported primitive in quadratic poly eval: div with operands {[type(u) for u in ins]}"
        )

    def _poly_concat(ins, eqn, P_in):
        polys = [u if isinstance(u, QuadPoly) else QuadPoly.to_poly_like(u, P_in) for u in ins]
        return QuadPoly.concat(polys, axis=int(eqn.params.get("dimension", 0)))

    def _poly_cumsum(ins, eqn, P_in):
        return _eval_poly_cumsum_arg(ins[0], eqn.params.get("axis", None), P_in)

    def _poly_reduce_sum(ins, eqn, P_in):
        del P_in
        return ins[0].reduce_sum(tuple(eqn.params.get("axes", ())))

    def _poly_jit(ins, eqn, P_in):
        inner = eqn.params.get("jaxpr", None)
        inner_eqns = getattr(getattr(inner, "jaxpr", None), "eqns", ())
        if (
            len(ins) == 1
            and len(inner_eqns) == 1
            and inner_eqns[0].primitive is jlax.cumsum_p
        ):
            return _eval_poly_cumsum_arg(ins[0], inner_eqns[0].params.get("axis", None), P_in)
        raise TypeError(
            f"Unsupported jit-wrapped primitive in quadratic poly eval: {eqn.params.get('name', '<unnamed>')}"
        )

    poly_handlers = {
        jlax.reshape_p: _poly_reshape,
        jlax.squeeze_p: _poly_squeeze,
        jlax.convert_element_type_p: _poly_convert,
        jlax.broadcast_in_dim_p: _poly_broadcast,
        jlax.slice_p: _poly_slice,
        jlax.add_p: _poly_add,
        jlax.sub_p: _poly_sub,
        jlax.mul_p: _poly_mul,
        jlax.neg_p: _poly_neg,
        jlax.sin_p: _poly_unary_handler(jnp.sin, jnp.cos, lambda x: -jnp.sin(x)),
        jlax.cos_p: _poly_unary_handler(jnp.cos, lambda x: -jnp.sin(x), lambda x: -jnp.cos(x)),
        jlax.tanh_p: _poly_unary_handler(jnp.tanh, lambda x: 1.0 - jnp.tanh(x)**2, lambda x: -2.0 * jnp.tanh(x) * (1.0 - jnp.tanh(x)**2)),
        jlax.tan_p: _poly_unary_handler(jnp.tan, lambda x: 1.0 / (jnp.cos(x) ** 2), lambda x: 2.0 * jnp.sin(x) / (jnp.cos(x) ** 3)),
        jlax.exp_p: _poly_unary_handler(jnp.exp, jnp.exp, jnp.exp),
        jlax.log_p: _poly_unary_handler(jnp.log, lambda x: 1.0 / x, lambda x: -1.0 / (x ** 2)),
        jlax.sqrt_p: _poly_unary_handler(jnp.sqrt, lambda x: 0.5 / jnp.sqrt(x), lambda x: -0.25 / (x ** (3/2))),
        jlax.div_p: _poly_div,
        jlax.concatenate_p: _poly_concat,
        jlax.cumsum_p: _poly_cumsum,
        jlax.reduce_sum_p: _poly_reduce_sum,
    }

    poly_name_handlers = {
        "stop_gradient": _poly_convert,
        "tie_in": _poly_convert,
        "jit": _poly_jit,
    }

    def eval_poly(x_poly: QuadPoly, step_lo: Array, step_hi: Array) -> QuadPoly:
        env: Dict[Any, Any] = {}
        # bind consts and invars (t, x)
        for v, c in zip(jaxpr.constvars, consts):
            env[v] = c
        x_var = jaxpr.invars[0]

        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            P_in = QuadPoly(x_poly.c, x_poly.L, jnp.zeros_like(x_poly.Lt))
        else:
            P_in = x_poly  # keep full degree-2 input poly
        env[x_var] = P_in

        for eqn in jaxpr.eqns:
            p = eqn.primitive
            pname = getattr(p, "name", "")
            ins = [env[i] if not hasattr(i, "val") else i.val for i in eqn.invars]
            has_symbolic = any(isinstance(u, (QuadPoly, QuadTM)) for u in ins)

            if not has_symbolic:
                out = p.bind(*ins, **eqn.params)
            elif p in poly_handlers:
                out = poly_handlers[p](ins, eqn, P_in)
            elif pname in poly_name_handlers:
                out = poly_name_handlers[pname](ins, eqn, P_in)
            else:
                raise TypeError(
                    f"Unsupported primitive in quadratic poly eval: {pname} with operands {[type(u) for u in ins]}"
                )

            if eqn.primitive.multiple_results and isinstance(out, (tuple, list)):
                for v, val in zip(eqn.outvars, out):
                    env[v] = val
            else:
                env[eqn.outvars[0]] = out

        out_poly: QuadPoly = env[jaxpr.outvars[0]]
        # out_poly = out_poly.append_zero_var(n = D - out_poly.L.shape[1])

        return out_poly
    return eval_poly

# ----------------------
# TM interpreter (quadratic version)
# ----------------------
def build_rhs_tm_from_prog(prog: ClosedProg, V: int) -> Callable[[QuadTM, Array, Array], QuadTM]:
    """Evaluate RHS into a QuadTM with reduced-order-2 keep for unary ops and Lagrange remainders."""
    jaxpr, consts, D = prog.jaxpr, prog.consts, prog.D

    def _broadcast_tm_to_shape(x: QuadTM, target_shape: tuple[int, ...]) -> QuadTM:
        if x.logical_shape == target_shape:
            return x
        if x.D == 1 and target_shape != (1,):
            return x.broadcast_in_dim(target_shape, ())
        if len(x.logical_shape) > len(target_shape):
            raise ValueError(f"Cannot broadcast shape {x.logical_shape} to {target_shape}")
        dims = tuple(range(len(target_shape) - len(x.logical_shape), len(target_shape)))
        return x.broadcast_in_dim(target_shape, dims)

    def _broadcast_tm_pair(a: QuadTM, b: QuadTM) -> tuple[QuadTM, QuadTM]:
        target_shape = jnp.broadcast_shapes(a.logical_shape, b.logical_shape)
        return _broadcast_tm_to_shape(a, target_shape), _broadcast_tm_to_shape(b, target_shape)

    def _eval_tm_cumsum_arg(arg, axis, ref_tm):
        if axis not in (0, None):
            raise NotImplementedError(f"Only axis=None/0 cumsum is supported, got axis={axis}")
        if isinstance(arg, (QuadTM, QuadPoly)):
            return QuadTM.to_tm_like(arg, ref_tm).cumsum(axis=None)
        return jnp.cumsum(arg, axis=axis)

    def _tm_reshape(ins, eqn, x_tm, box_lo, box_hi):
        del x_tm, box_lo, box_hi
        return ins[0].reshape(tuple(eqn.params["new_sizes"]))

    def _tm_squeeze(ins, eqn, x_tm, box_lo, box_hi):
        del x_tm, box_lo, box_hi
        return ins[0].squeeze(tuple(eqn.params.get("dimensions", ())))

    def _tm_convert(ins, eqn, x_tm, box_lo, box_hi):
        del eqn, box_lo, box_hi
        return ins[0] if isinstance(ins[0], QuadTM) else QuadTM.to_tm_like(ins[0], x_tm)

    def _tm_broadcast(ins, eqn, x_tm, box_lo, box_hi):
        del box_lo, box_hi
        broadcast_shape = tuple(eqn.params["shape"])
        if isinstance(ins[0], QuadTM):
            return ins[0].broadcast_in_dim(broadcast_shape, tuple(eqn.params.get("broadcast_dimensions", ())))
        if broadcast_shape == (1,):
            return QuadTM.to_tm_like(ins[0], x_tm)
        return QuadTM.to_tm_like(jnp.broadcast_to(ins[0], broadcast_shape), x_tm)

    def _tm_slice(ins, eqn, x_tm, box_lo, box_hi):
        del x_tm, box_lo, box_hi
        starts  = tuple(eqn.params.get("start_indices", ()))
        limits  = tuple(eqn.params.get("limit_indices", ()))
        strides = eqn.params.get("strides", None) or (1,)
        return QuadTM.slice(ins[0], starts if len(starts) > 1 else int(starts[0]), limits if len(limits) > 1 else int(limits[0]), strides)

    def _tm_add(ins, eqn, x_tm, box_lo, box_hi):
        del eqn, box_lo, box_hi
        A, B = _broadcast_tm_pair(QuadTM.to_tm_like(ins[0], x_tm), QuadTM.to_tm_like(ins[1], x_tm))
        return A.add(B)

    def _tm_sub(ins, eqn, x_tm, box_lo, box_hi):
        del eqn, box_lo, box_hi
        A, B = _broadcast_tm_pair(QuadTM.to_tm_like(ins[0], x_tm), QuadTM.to_tm_like(ins[1], x_tm))
        return A.sub(B)

    def _tm_neg(ins, eqn, x_tm, box_lo, box_hi):
        del eqn, box_lo, box_hi
        a = ins[0]
        if isinstance(a, QuadTM):
            return QuadTM(a.P.scale(-1.0), Interval(-a.R.hi, -a.R.lo))
        if isinstance(a, QuadPoly):
            return QuadTM.from_poly(a.scale(-1.0))
        return QuadTM.to_tm_like(a, x_tm).scale(-1.0)

    def _tm_mul(ins, eqn, x_tm, box_lo, box_hi):
        del eqn
        A, B = _broadcast_tm_pair(QuadTM.to_tm_like(ins[0], x_tm), QuadTM.to_tm_like(ins[1], x_tm))
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            return A._mul_ctrunc1(B, box_lo, box_hi)
        return A.mul(B, box_lo, box_hi)

    def _tm_unary_handler(fn, fp, fpp, iv_fn, iv_fpp, iv_fppp):
        def _handler(ins, eqn, x_tm, box_lo, box_hi):
            del eqn
            A = QuadTM.to_tm_like(ins[0], x_tm)
            return tm_unary_lagrange(A, fn, fp, fpp, box_lo, box_hi, iv_fn, iv_fpp, iv_fppp)
        return _handler

    def _tm_div(ins, eqn, x_tm, box_lo, box_hi):
        del eqn
        a, b = ins
        if isinstance(b, (QuadTM, QuadPoly)):
            A = QuadTM.to_tm_like(a, x_tm)
            B = QuadTM.to_tm_like(b, x_tm)
            return A.div(B, box_lo, box_hi)
        if isinstance(a, (QuadTM, QuadPoly)):
            return QuadTM.to_tm_like(a, x_tm).scale(1.0 / b)
        raise TypeError(
            f"Unsupported primitive in quadratic TM eval: div with operands {[type(u) for u in ins]}"
        )

    def _tm_concat(ins, eqn, x_tm, box_lo, box_hi):
        del box_lo, box_hi
        tms = [u if isinstance(u, QuadTM) else QuadTM.to_tm_like(u, x_tm) for u in ins]
        return QuadTM.concat(tms, axis=int(eqn.params.get("dimension", 0)))

    def _tm_cumsum(ins, eqn, x_tm, box_lo, box_hi):
        del box_lo, box_hi
        return _eval_tm_cumsum_arg(ins[0], eqn.params.get("axis", None), x_tm)

    def _tm_reduce_sum(ins, eqn, x_tm, box_lo, box_hi):
        del x_tm, box_lo, box_hi
        return ins[0].reduce_sum(tuple(eqn.params.get("axes", ())))

    def _tm_jit(ins, eqn, x_tm, box_lo, box_hi):
        del box_lo, box_hi
        inner = eqn.params.get("jaxpr", None)
        inner_eqns = getattr(getattr(inner, "jaxpr", None), "eqns", ())
        if (
            len(ins) == 1
            and len(inner_eqns) == 1
            and inner_eqns[0].primitive is jlax.cumsum_p
        ):
            return _eval_tm_cumsum_arg(ins[0], inner_eqns[0].params.get("axis", None), x_tm)
        raise TypeError(
            f"Unsupported jit-wrapped primitive in quadratic TM eval: {eqn.params.get('name', '<unnamed>')}"
        )

    tm_handlers = {
        jlax.reshape_p: _tm_reshape,
        jlax.squeeze_p: _tm_squeeze,
        jlax.convert_element_type_p: _tm_convert,
        jlax.broadcast_in_dim_p: _tm_broadcast,
        jlax.slice_p: _tm_slice,
        jlax.add_p: _tm_add,
        jlax.sub_p: _tm_sub,
        jlax.neg_p: _tm_neg,
        jlax.mul_p: _tm_mul,
        jlax.sin_p: _tm_unary_handler(jnp.sin, jnp.cos, lambda x: -jnp.sin(x), iv_sin, iv_sin_fpp, iv_sin_fppp),
        jlax.cos_p: _tm_unary_handler(jnp.cos, lambda x: -jnp.sin(x), lambda x: -jnp.cos(x), iv_cos, iv_cos_fpp, iv_cos_fppp),
        jlax.tanh_p: _tm_unary_handler(jnp.tanh, lambda x: 1.0 - jnp.tanh(x)**2, lambda x: -2.0*jnp.tanh(x)*(1.0 - jnp.tanh(x)**2), iv_tanh, iv_tanh_fpp, iv_tanh_fppp),
        jlax.tan_p: _tm_unary_handler(jnp.tan, lambda x: 1.0 / (jnp.cos(x) ** 2), lambda x: 2.0*jnp.tan(x) / (jnp.cos(x) ** 2), iv_tan, iv_tan_fpp, iv_tan_fppp),
        jlax.exp_p: _tm_unary_handler(jnp.exp, jnp.exp, jnp.exp, iv_exp, iv_exp_fpp, iv_exp_fppp),
        jlax.log_p: _tm_unary_handler(jnp.log, lambda x: 1.0 / x, lambda x: -1.0 / (x * x), iv_log, iv_log_fpp, iv_log_fppp),
        jlax.sqrt_p: _tm_unary_handler(jnp.sqrt, lambda x: 0.5 / jnp.sqrt(x), lambda x: -0.25 / (x ** 1.5), iv_sqrt, iv_sqrt_fpp, iv_sqrt_fppp),
        jlax.div_p: _tm_div,
        jlax.concatenate_p: _tm_concat,
        jlax.cumsum_p: _tm_cumsum,
        jlax.reduce_sum_p: _tm_reduce_sum,
    }

    tm_name_handlers = {
        "stop_gradient": _tm_convert,
        "tie_in": _tm_convert,
        "jit": _tm_jit,
    }

    def eval_tm(x_tm: QuadTM, box_lo: Array, box_hi: Array) -> QuadTM:
        # x_tm.log("Entering TM eval")
        env: Dict[Any, Any] = {}

        for v, c in zip(jaxpr.constvars, consts):
            env[v] = c
        x_var = jaxpr.invars[0]
        if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            env[x_var] = x_tm.truncate_to_affine(box_lo, box_hi)  # enforce degree-1 on input TM
        else:
            env[x_var] = x_tm  # keep full degree-2 input TM

        for eqn in jaxpr.eqns:
            p = eqn.primitive
            pname = getattr(p, "name", "")
            ins = [env[i] if not hasattr(i, "val") else i.val for i in eqn.invars]

            has_symbolic = any(isinstance(u, (QuadTM, QuadPoly)) for u in ins)

            if not has_symbolic:
                out = p.bind(*ins, **eqn.params)
            elif p in tm_handlers:
                out = tm_handlers[p](ins, eqn, x_tm, box_lo, box_hi)
            elif pname in tm_name_handlers:
                out = tm_name_handlers[pname](ins, eqn, x_tm, box_lo, box_hi)
            else:
                raise TypeError(
                    f"Unsupported primitive in quadratic TM eval: {pname} with operands {[type(u) for u in ins]}"
                )

            if eqn.primitive.multiple_results and isinstance(out, (tuple, list)):
                raise NotImplementedError("Multiple results not expected in current RHS forms")
            else:
                env[eqn.outvars[0]] = out

        out_tm = env[jaxpr.outvars[0]]
        # out_tm = out_tm.append_zero_var(n = D - out_tm.P.c.shape[1])

        return out_tm

    return eval_tm


# ======================
# Convenience wrapper
# ======================
def build_auto_rhs_analytic(f: Callable, D: int, V: int) -> Tuple[
    Callable[[QuadTM], QuadPoly],
    Callable[[QuadTM, Array, Array], QuadTM]
]:
    prog = _trace_prog(f, D)
    return build_rhs_poly_from_prog(prog, V), build_rhs_tm_from_prog(prog, V)
