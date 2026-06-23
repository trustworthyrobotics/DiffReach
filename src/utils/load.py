# src/utils.py
from __future__ import annotations
import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import onnx
import yaml
import jax.numpy as jnp
from jaxonnxruntime.backend import Backend as ONNXJaxBackend
from jaxonnxruntime.core import config_class
config = config_class.config
config.update("jaxort_only_allow_initializers_as_static_args", False)

# ----------------------------
# Config / Loading
# ----------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_analytic_dynamics(py_path: str, D_s: int, D_a: int, discrete: bool, fn_name: str = "dynamics") -> Callable[[jnp.ndarray], jnp.ndarray]:
    spec = importlib.util.spec_from_file_location("dyn_mod", py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import dynamics module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if not hasattr(mod, fn_name):
        raise AttributeError(f"{py_path} does not define a function `{fn_name}`")
    dyn = getattr(mod, fn_name)
    def f(x):
        x_out = dyn(x)
        u_out = jnp.zeros_like(x[..., D_s:])
        return jnp.concatenate([x_out, u_out], axis=0)

    def f_d(x):
        x_out = dyn(x)
        u_out = x[..., D_s:]
        return jnp.concatenate([x_out, u_out], axis=0)
    return f if not discrete else f_d

def load_nn_dynamics(nn_path: str, D_s: int, D_a: int, discrete: bool) -> Callable[[jnp.ndarray], jnp.ndarray]:
    nn_path = Path(nn_path)
    model=onnx.load_model(nn_path.with_suffix(".onnx"))
    in_names=[vi.name for vi in model.graph.input]
    in_len=model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
    out_len=model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
    assert in_len == D_s + D_a, f"Input dimension mismatch: model expects {in_len}, but D_s + D_a = {D_s + D_a}"
    assert out_len == D_s, f"Output dimension mismatch: model expects {out_len}, but D_s = {D_s}"
    backend=ONNXJaxBackend.prepare(model)
    def f(x):
        dx = backend.run({in_names[0]: x})[0]
        du = jnp.zeros_like(x[..., D_s:])
        return jnp.concatenate([dx, du], axis=0)
    def f_d(x):
        dx = backend.run({in_names[0]: x})[0]
        du = x[..., D_s:]
        return jnp.concatenate([dx, du], axis=0)
    return f if not discrete else f_d

def load_controller(controller_path: str):
    controller_path = Path(controller_path)
    model=onnx.load_model(controller_path.with_suffix(".onnx"))
    in_names=[vi.name for vi in model.graph.input]
    backend=ONNXJaxBackend.prepare(model)
    def f(x):
        return backend.run({in_names[0]: x})[0]
    return f
