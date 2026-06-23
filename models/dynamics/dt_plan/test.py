import jax.numpy as jnp


def dynamics(x):
	x1, x2, u1, u2 = [x[i] for i in range(4)]
	# dx1 = jnp.cos(x1)
	# dx2 = jnp.sin(x2)
	x1_next = x1 + jnp.sin(x1) - u2
	x2_next = x2 + u1
	return jnp.stack([x1_next, x2_next], axis=0)


import os
import warnings

warnings.filterwarnings("ignore", module="jax2onnx.plugins")

import jax

# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx

from jax2onnx import to_onnx
import onnx


class MLP(eqx.Module):
	layers: list
	in_size: int
	hidden_size: int
	out_size: int
	residual: bool

	def __init__(self, in_size, hidden_size, out_size, residual,key):
		k1, k2 = jax.random.split(key)
		self.layers = [
			eqx.nn.Linear(in_size, hidden_size, key=k1),
			eqx.nn.Linear(hidden_size, out_size, key=k2),
		]
		self.in_size = in_size
		self.hidden_size = hidden_size
		self.out_size = out_size
		self.residual = residual

	def forward(self, x):
		z = jax.nn.relu(self.layers[0](x))
		y = self.layers[1](z)
		return y

	def forward_residual(self, x):
		z = jax.nn.relu(self.layers[0](x))
		y = self.layers[1](z) + x[: self.out_size]
		return y

	def __call__(self, x):
		# return self.layers[0](x)
		if self.residual:
			return self.forward_residual(x)
		else:
			return self.forward(x)	



def create_model(in_size, hidden_size, out_size, residual, onnx_model_path):
	key = jax.random.PRNGKey(0)
	model = MLP(in_size, hidden_size, out_size, residual, key)
	random_input = jax.random.normal(key, (1, in_size))
	model_output = jax.vmap(model)(random_input)
	print("Model output:", model_output)

	onnx_model = to_onnx(
		model,
		[(in_size, )],
		# return_mode="file",
		# output_path=onnx_model_path,
		opset=19,
	)
	onnx.save(onnx_model, onnx_model_path)
	# print onnx model summary
	print(onnx.helper.printable_graph(onnx_model.graph))
	return

if __name__ == "__main__":
	key = jax.random.PRNGKey(0)
	state_dim = 2
	action_dim = 2
	cur_file_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
	model_name = "test.onnx"
	onnx_model_path = os.path.join(cur_file_dir, model_name)
	create_model(state_dim + action_dim, 64, state_dim, True, onnx_model_path)
