import jax.numpy as jnp

def dynamics(x):
    x1, x2 = [x[i] for i in range(2)]
    # dx1 = jnp.cos(x1)
    # dx2 = jnp.sin(x2)
    x1_next = x1 + jnp.sin(x1)
    x2_next = x2
    return jnp.stack([x1_next, x2_next], axis=0)

import os
import warnings
warnings.filterwarnings("ignore", module="jax2onnx.plugins")

import jax
import jax.numpy as jnp
import equinox as eqx

from jax2onnx import to_onnx

class MLP(eqx.Module):
    layers: list

    def __init__(self, in_size, hidden_size, out_size, key):
        k1, k2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(in_size, hidden_size, key=k1),
            eqx.nn.Linear(hidden_size, out_size, key=k2)
        ]

    def __call__(self, x):
        # return self.layers[0](x)
        z = jax.nn.relu(self.layers[0](x))
        y = self.layers[1](z) + x
        return y

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    in_size = 2
    hidden_size = 64
    out_size = 2
    model = MLP(in_size, hidden_size, out_size, key)
    random_input = jax.random.normal(key, (in_size,))
    model_output = model(random_input)
    print("Model output:", model_output)

    cur_file_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    onnx_model_path = os.path.join(cur_file_dir, "test.onnx")
    to_onnx(
        model,
        [("B", in_size)],
        return_mode="file",
        output_path=onnx_model_path,
    )
