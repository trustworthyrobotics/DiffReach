import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
import onnx
from onnx import numpy_helper

# NEW
def merge_matmul_add_to_gemm_inplace(model):
    """
    In-place: MatMul[(A, B)] [+ Add(MatMul_out, C)]  ->  Gemm(A, B[, C])
    - Handles optional Transpose on B with perm=(1,0) by setting transB=1.
    - Lifts Constant tensors to initializers.
    - Skips cases with Transpose on A (transA=1), to stay compatible with downstream.
    """
    import onnx
    from onnx import helper, numpy_helper

    g = model.graph

    # --- Maps ---
    init_map = {init.name: init for init in g.initializer}
    output_to_node = {out: n for n in g.node for out in n.output}
    consumers = {}
    for n in g.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)

    # Collect Constant tensors (value attr) → numpy arrays keyed by node output name
    const_np = {}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    t = onnx.helper.get_attribute_value(a)  # TensorProto
                    const_np[n.output[0]] = numpy_helper.to_array(t)

    def ensure_init(name_hint, arr_np):
        if name_hint in init_map:
            return name_hint
        new_name = name_hint
        # avoid collisions
        i = 0
        while new_name in init_map:
            i += 1
            new_name = f"{name_hint}__{i}"
        tensor = numpy_helper.from_array(arr_np, name=new_name)
        g.initializer.append(tensor)
        init_map[new_name] = tensor
        return new_name

    def resolve_weight_or_bias(inp_name):
        # Already an initializer?
        if inp_name in init_map:
            return inp_name
        # Constant node output?
        if inp_name in const_np:
            return ensure_init(f"const_init__{inp_name}", const_np[inp_name])
        return None  # not resolvable to a static tensor

    def get_attr_dict(node):
        return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

    new_nodes = []
    skip_ids = set()
    node_id = lambda n: id(n)

    for n in g.node:
        if node_id(n) in skip_ids or n.op_type != "MatMul":
            if node_id(n) not in skip_ids:
                new_nodes.append(n)
            continue

        mm = n
        a_in, b_in = mm.input[0], mm.input[1]
        transA = 0  # we won't support folding Transpose on A → keep 0
        transB = 0

        # If B is from Transpose(..., perm=[1,0]), fold by using its input and setting transB=1
        if b_in in output_to_node and output_to_node[b_in].op_type == "Transpose":
            tnode = output_to_node[b_in]
            perm = get_attr_dict(tnode).get("perm", None)
            if perm in (None, [1, 0], (1, 0)):  # default perm=[1,0] for 2D if omitted
                b_in = tnode.input[0]
                transB = 1
                # delete transpose if only used by this MatMul
                if len(consumers.get(tnode.output[0], [])) == 1:
                    skip_ids.add(node_id(tnode))
            else:
                # unsupported transpose; leave as-is
                new_nodes.append(mm)
                continue

        # If A is a Transpose, we'd need transA=1 (downstream converter rejects that).
        if a_in in output_to_node and output_to_node[a_in].op_type == "Transpose":
            # Skip merging to keep downstream behavior predictable.
            new_nodes.append(mm)
            continue

        b_name = resolve_weight_or_bias(b_in)
        if b_name is None:
            new_nodes.append(mm)
            continue

        # Find optional Add consumer using MatMul's output
        mm_out = mm.output[0]
        add = None
        for c in consumers.get(mm_out, []):
            if c.op_type == "Add":
                add = c
                break

        # Resolve optional bias C
        c_name = None
        gemm_out = mm_out
        if add is not None:
            # pick the other input as bias
            add_inps = list(add.input)
            add_inps.remove(mm_out)
            bias_in = add_inps[0]
            c_name = resolve_weight_or_bias(bias_in)
            if c_name is None:
                # Cannot fold bias; still can fold MatMul→Gemm without C
                pass
            gemm_out = add.output[0]

        # Build Gemm
        inputs = [a_in, b_name] + ([c_name] if c_name is not None else [])
        gemm = helper.make_node(
            "Gemm",
            inputs=inputs,
            outputs=[gemm_out],
            name=(mm.name or "MatMul") + "_fused_gemm",
            alpha=1.0,
            beta=1.0,
            transA=transA,
            transB=transB,
        )

        new_nodes.append(gemm)
        skip_ids.add(node_id(mm))
        if add is not None:
            skip_ids.add(node_id(add))

    # Replace graph nodes
    g.ClearField("node")
    g.node.extend(new_nodes)
    return model



# NEW: utility maps and helpers for richer MLP support
_ACTS = {
    "Relu": (lambda jax, jnp: jax.nn.relu),
    "Tanh": (lambda jax, jnp: jnp.tanh),
    "Sigmoid": (lambda jax, jnp: jax.nn.sigmoid),
    "Softplus": (lambda jax, jnp: jax.nn.softplus),
    "Softsign": (lambda jax, jnp: jax.nn.soft_sign if hasattr(jax.nn, "soft_sign") else (lambda x: x / (1 + jnp.abs(x)))),
    "Elu": (lambda jax, jnp: jax.nn.elu),
    "LeakyRelu": (lambda jax, jnp: lambda x, _alpha=None: jnp.where(x >= 0, x, (_alpha if _alpha is not None else 0.01) * x)),
    "Gelu": (lambda jax, jnp: jax.nn.gelu),
    # ONNX HardSigmoid(alpha,beta): y = clip(alpha*x + beta, 0,1)
    "HardSigmoid": (lambda jax, jnp: None),  # handled via Clip+Mul+Add if needed
}

def _attrs(node):
    import onnx
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

def _lift_constants_to_initializers(model):
    "Turn Constant nodes into graph.initializer tensors (if not already)."
    import onnx
    from onnx import numpy_helper
    g = model.graph
    init_names = {init.name for init in g.initializer}
    new_inits = []
    for n in list(g.node):
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    t = onnx.helper.get_attribute_value(a)
                    name = n.output[0]
                    if name not in init_names:
                        new_inits.append(numpy_helper.from_array(numpy_helper.to_array(t), name=name))
    g.initializer.extend(new_inits)
    return model

def _initializers_numpy(graph):
    from onnx import numpy_helper
    return {init.name: numpy_helper.to_array(init) for init in graph.initializer}

def _ensure_bias_vector(b, out_features, jnp):
    if b is None:
        return jnp.zeros((out_features,), dtype=jnp.float32)
    b = jnp.asarray(b)
    if b.ndim == 0:
        return jnp.full((out_features,), b, dtype=b.dtype)
    b = b.reshape(-1)
    if b.size == out_features:
        return b
    if b.size == out_features * 1:
        return b.reshape(out_features, -1).squeeze(-1)
    raise ValueError(f"Unexpected bias size {b.size}; expected {out_features}.")

def _make_linear_from_gemm(node, inits, jax, jnp, nn, eqx):
    "Create an Equinox Linear from a Gemm node and return (module, out_features, in_features)."
    a = _attrs(node)
    alpha = float(a.get("alpha", 1.0))
    beta = float(a.get("beta", 1.0))
    transA = int(a.get("transA", 0))
    transB = int(a.get("transB", 0))
    if transA:
        raise ValueError("Gemm with transA=1 not supported.")

    B_name = node.input[1]
    C_name = node.input[2] if len(node.input) >= 3 else None
    B = inits[B_name]
    C = inits[C_name] if C_name is not None else None

    W = (B if transB == 1 else B.T) * alpha
    out_features, in_features = W.shape
    W = jnp.asarray(W, dtype=jnp.float32)
    b = _ensure_bias_vector(C * beta if C is not None else None, out_features, jnp)

    lin = nn.Linear(in_features, out_features, key=jax.random.PRNGKey(0), use_bias=True)
    lin = eqx.tree_at(lambda l: l.weight, lin, W)
    lin = eqx.tree_at(lambda l: l.bias, lin, jnp.asarray(b, dtype=jnp.float32))
    return lin, out_features, in_features

def _maybe_activation(node, nn, jax, jnp):
    "Return nn.Lambda for activation or None."
    if node.op_type in _ACTS:
        fn_maker = _ACTS[node.op_type]
        if node.op_type == "LeakyRelu":
            alpha = float(_attrs(node).get("alpha", 0.01))
            return nn.Lambda(lambda x: jnp.where(x >= 0, x, alpha * x))
        if node.op_type == "HardSigmoid":
            a = _attrs(node)
            alpha = float(a.get("alpha", 0.2))
            beta = float(a.get("beta", 0.5))
            return nn.Lambda(lambda x: jnp.clip(alpha * x + beta, 0.0, 1.0))
        return nn.Lambda(fn_maker(jax, jnp))
    return None

def _maybe_elementwise(node, inits, nn, jnp):
    "Handle simple Add/Mul with scalar or feature-wise constant."
    if node.op_type not in ("Add", "Sub", "Mul"):
        return None
    # one input should be constant/initializer
    const = None
    for name in node.input:
        if name in inits:
            const = inits[name]
            break
    if const is None:
        return None  # non-constant residual add not handled here
    const = jnp.asarray(const)
    if const.ndim == 0:
        c = float(const)
        return nn.Lambda((lambda x, c=c: x + c) if node.op_type == "Add" else (lambda x, c=c: x * c))
    # Allow feature-wise vector broadcast on last dim; we implement as Lambda
    def f(x, c=const, is_add=(node.op_type == "Add")):
        # works for (B,out) or (out, B) or (out,)
        if x.ndim == 2:
            if x.shape[1] == c.shape[0]:   # (B,out)
                return x + c if is_add else x * c
            if x.shape[0] == c.shape[0]:   # (out,B)
                return (x.T + c if is_add else x.T * c).T
        if x.ndim == 1 and x.shape[0] == c.shape[0]:
            return x + c if is_add else x * c
        # fallback: try to broadcast on last axis
        return (x + c) if is_add else (x * c)
    return nn.Lambda(f)

def _maybe_flatten_like(node, nn, jnp):
    "Return a flatten Lambda if node is Flatten(axis=1) or Reshape to (B,-1)."
    if node.op_type == "Flatten":
        axis = int(_attrs(node).get("axis", 1))
        if axis != 1:
            raise ValueError("Only Flatten(axis=1) supported.")
        return nn.Lambda(lambda x: x.reshape(x.shape[0], -1) if x.ndim >= 2 else x)
    if node.op_type == "Reshape":
        # Treat any reshape to (B,-1) as flatten
        # We can't access the constant shape easily here; handled by keeping generic behavior
        return nn.Lambda(lambda x: x.reshape(x.shape[0], -1) if x.ndim >= 2 else x)
    return None


# NEW
def _make_conv2d_from_onnx(node, inits, jax, jnp, nn, eqx):
    """
    Create an Equinox nn.Conv2d from an ONNX Conv node.
    Assumes NCHW data format. Supports groups, symmetric padding, per-axis stride/dilation.
    Weight layout (ONNX): (out_c, in_c/group, kH, kW)
    """
    a = _attrs(node)
    strides   = tuple(a.get("strides", [1, 1]))
    dilations = tuple(a.get("dilations", [1, 1]))
    pads      = list(a.get("pads", [0, 0, 0, 0]))  # [top,left,bottom,right]
    groups    = int(a.get("group", 1))

    if len(strides) != 2 or len(dilations) != 2:
        raise ValueError("Only 2D Conv is supported.")
    if len(pads) == 4:
        pad_h = int(pads[0]); pad_w = int(pads[1])
        if pads[0] != pads[2] or pads[1] != pads[3]:
            raise ValueError("Asymmetric padding not supported.")
        padding = (pad_h, pad_w)
    else:
        padding = (0, 0)

    W_name = node.input[1]
    B_name = node.input[2] if len(node.input) >= 3 else None
    if W_name not in inits:
        raise ValueError("Conv weights must be an initializer.")
    W = jnp.asarray(inits[W_name], dtype=jnp.float32)  # (out_c, in_c/groups, kH, kW)

    out_c, in_c_per_g, kH, kW = W.shape
    in_c = in_c_per_g * groups
    bias = None
    if B_name is not None:
        if B_name not in inits:
            raise ValueError("Conv bias must be an initializer when present.")
        bias = jnp.asarray(inits[B_name], dtype=jnp.float32).reshape(-1)
        if bias.shape[0] != out_c:
            raise ValueError(f"Bias/out_channels mismatch: {bias.shape[0]} vs {out_c}")

    conv = nn.Conv2d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=(kH, kW),
        stride=strides,
        padding=padding,
        dilation=dilations,
        groups=groups,
        use_bias=bias is not None,
        key=jax.random.PRNGKey(0),
    )
    conv = eqx.tree_at(lambda c: c.weight, conv, W)
    if bias is not None:
        conv = eqx.tree_at(lambda c: c.bias, conv, bias)
    return conv, (in_c, out_c, (kH, kW))


def convert_onnx_to_eqx(onnx_path: str) -> str:
    """
    Refactored converter with broader MLP support.
    Supported ops (any order along the main path):
      - Flatten(axis=1)/Reshape-to-(B,-1)
      - Gemm (preferred); MatMul(+Transpose on B) and Add are pre-merged to Gemm by merge_matmul_add_to_gemm_inplace().
      - Activations: Relu, Tanh, Sigmoid, Softplus, Softsign, Elu, LeakyRelu(alpha), Gelu, HardSigmoid(alpha,beta)
      - Elementwise Add/Mul with scalar or feature-wise constant vectors
      - Dropout/Cast/Identity: ignored (no-ops)
    Assumes a single-input single-output MLP (no convolutions/branches).

    Returns: path to the saved .eqx
    """
    from pathlib import Path
    import onnx

    # 1) Load and normalise graph
    model = onnx.load(onnx_path)
    model = _lift_constants_to_initializers(model)
    model = merge_matmul_add_to_gemm_inplace(model)  # fuse MatMul(+Add) → Gemm
    g = model.graph
    inits = _initializers_numpy(g)

    # 2) Build a flexible nn.Sequential by walking nodes linearly.
    mods = []

    # Optional flatten/reshape at the front (we'll also accept flatten anywhere harmlessly)
    for idx, node in enumerate(g.node):
        if node.op_type in ("Flatten", "Reshape"):
            m = _maybe_flatten_like(node, nn, jnp)
            if m is not None:
                mods.append(m)
            else:
                # Unsupported variant of flatten/reshape
                pass
        elif node.op_type in ("Identity", "Cast", "Dropout"):
            # No-op for serialization
            pass
        elif node.op_type == "Gemm":
            lin, out_f, in_f = _make_linear_from_gemm(node, inits, jax, jnp, nn, eqx)
            mods.append(lin)
        elif node.op_type == "Conv":
            conv, (in_c, out_c, (kH, kW)) = _make_conv2d_from_onnx(node, inits, jax, jnp, nn, eqx)
            mods.append(conv)
        elif node.op_type in _ACTS:
            act = _maybe_activation(node, nn, jax, jnp)
            if act is not None:
                mods.append(act)
        elif node.op_type in ("Add", "Sub", "Mul"):
            elem = _maybe_elementwise(node, inits, nn, jnp)
            if elem is not None:
                mods.append(elem)
            else:
                # Non-constant residual Add not supported in this minimal MLP converter
                raise ValueError("Encountered non-constant Add/Mul; residual branches are unsupported.")
        else:
            raise ValueError(f"Unsupported op in MLP path: {node.op_type}")

    seq = nn.Sequential(mods)
    out_path = str(Path(onnx_path).with_suffix(".eqx"))
    eqx.tree_serialise_leaves(out_path, seq)
    return out_path

if __name__ == "__main__":
    # assert len(sys.argv) == 2, "Usage: python convert.py model.onnx"
    # onnx_path = sys.argv[1]
    # out_path = convert_onnx_to_eqx(onnx_path)
    # print(f"Converted {onnx_path} -> {out_path}")
    convert_directory = "ARCH-COMP2024/benchmarks"
    # recursively convert all .onnx files in the directory
    for onnx_path in Path(convert_directory).rglob("*.onnx"):
        try:
            out_path = convert_onnx_to_eqx(str(onnx_path))
            # print(f"Converted {onnx_path} -> {out_path}")
        except Exception as e:
            print(f"Failed to convert {onnx_path}: {e}")