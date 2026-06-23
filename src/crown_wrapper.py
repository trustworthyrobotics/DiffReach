from collections import namedtuple
from pathlib import Path
from typing import Callable, Literal, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax_verify as jv

from src.taylor_model import Interval
import src.settings as settings

dtype = jnp.float32 if not settings.CONFIG["FP64_IN_CROWN"] else jnp.float64

""" 
code adapted from immrax.
"""

bound_propagation = jv.src.bound_propagation
bound_utils = jv.src.bound_utils
concretization = jv.src.concretization
synthetic_primitives = jv.src.synthetic_primitives
backward_crown = jv.src.linear.backward_crown
linear_relaxations = jv.src.linear.linear_relaxations


class LinFunExtractionConcretizer(concretization.BackwardConcretizer):
    """Linear function extractor.

    Given an objective over an output, extract the corresponding linear
    function over a target node.
    The relation between the output node and the target node are obtained by
    propagating backward the `base_transform`.
    """

    def __init__(self, base_transform, target_index, obj):
        self._base_transform = base_transform
        self._target_index = target_index
        self._obj = obj

    def should_handle_as_subgraph(self, primitive):
        return self._base_transform.should_handle_as_subgraph(primitive)

    def concretize_args(self, primitive):
        return self._base_transform.concretize_args(primitive)

    def concrete_bound(self, graph, inputs, env, node_ref):
        initial_lin_expression = linear_relaxations.LinearExpression(
            self._obj, jnp.zeros(self._obj.shape[:1])
        )
        target_linfun, _ = graph.backward_propagation(
            self._base_transform,
            env,
            {node_ref: initial_lin_expression},
            [self._target_index],
        )
        return target_linfun


class CROWNResult(namedtuple("CROWNResult", ["lC", "uC", "ld", "ud"])):
    def __call__(self, x: Union[jax.Array, Interval]) -> Interval:
        if isinstance(x, Interval):
            lCp = jnp.clip(self.lC, 0, jnp.inf)
            lCn = jnp.clip(self.lC, -jnp.inf, 0)
            uCp = jnp.clip(self.uC, 0, jnp.inf)
            uCn = jnp.clip(self.uC, -jnp.inf, 0)
            lo = x.lo[..., None]
            hi = x.hi[..., None]
            return Interval(
                (lCp @ lo + lCn @ hi).squeeze(-1) + self.ld,
                (uCn @ lo + uCp @ hi).squeeze(-1) + self.ud,
            )
        elif isinstance(x, jax.Array):
            x = x[..., None]
            return Interval((self.lC @ x).squeeze(-1) + self.ld, (self.uC @ x).squeeze(-1) + self.ud)


def crown(
    f: Callable[..., jax.Array], in_len:int, out_len: int, enable_r:bool = False
) -> Callable[..., CROWNResult]:

    obj = (jnp.vstack([jnp.eye(out_len, dtype=dtype), -jnp.eye(out_len, dtype=dtype)]))

    def F(init_bound, affine_weight: jax.Array = jnp.eye(in_len, dtype=dtype), affine_coeff: jax.Array = jnp.zeros((in_len,), dtype=dtype) ) -> CROWNResult:
        """Run CROWN but return linfuns rather than concretized IntervalBounds.

        Parameters
        ----------
        bound :
            Bounds on the inputs of the function.

        Returns
        -------


        """

        bound = jv.IntervalBound(init_bound.lo, init_bound.hi)
        def f_wrapped_wo_r(x):
            x = jnp.dot(affine_weight, x) + affine_coeff
            return f(x)

        def f_wrapped_r(x):
            x_ = x[:in_len]
            r = x[in_len:]
            x = jnp.dot(affine_weight, x_) + affine_coeff + r
            return f(x)

        f_wrapped = f_wrapped_r if enable_r else f_wrapped_wo_r

        # As we want to extract some linfuns that are in the middle of two linear
        # layers, we want to avoid the linear operator fusion.
        simplifier_composition = synthetic_primitives.simplifier_composition
        default_simplifier_without_linear = simplifier_composition(
            synthetic_primitives.activation_simplifier,
            synthetic_primitives.hoist_constant_computations,
        )

        # We are first going to obtain intermediate bounds for all the activation
        # of the network, so that the backward propagation of the extraction can be
        # done.
        bound_retriever_algorithm = bound_utils.BoundRetrieverAlgorithm(
            concretization.BackwardConcretizingAlgorithm(
                backward_crown.backward_crown_concretizer
            )
        )
        # BoundRetrieverAlgorithm wraps an existing algorithm and captures all of
        # the intermediate bound it generates.
        bound_propagation.bound_propagation(
            bound_retriever_algorithm,
            f_wrapped,
            bound,
            graph_simplifier=default_simplifier_without_linear,
        )
        intermediate_bounds = bound_retriever_algorithm.concrete_bounds
        # Now that we have extracted all intermediate bounds, we create a
        # FixedBoundApplier. This is a forward transform that pretends to compute
        # bounds, but actually just look them up in a dict of precomputed bounds.
        fwd_bound_applier = bound_utils.FixedBoundApplier(intermediate_bounds)

        # # Let's define what node we are interested in capturing linear functions
        # # for. If needed, this could be extracted and given as argument to this
        # # function, or as a callback that would compute which nodes to target.
        # input_indices = [(i,) for i, _ in enumerate(bounds)]
        # # We're propagating to the first input.
        # target_index = input_indices[0]

        target_index = (0,)

        # Create the concretizer. See the class definition above. The definition
        # of a "concretized_bound" for this one is "Obj linear function
        # reformulated as a linear function of target index".
        # Note: If there is a need to handle a network with multiple output, it
        # should be easy to extend by making obj here a dict mapping output node to
        # objective on that output node.
        extracting_concretizer = LinFunExtractionConcretizer(
            backward_crown.backward_crown_transform, target_index, obj
        )
        # BackwardAlgorithmForwardConcretization uses:
        #  - A forward transform to compute all intermediate bounds (here a bound
        #    applier that just look them up).
        #  - A backward concretizer to compute all final bounds (which we have here
        #    defined as the linear function of the target index).
        fwd_bwd_alg = concretization.BackwardAlgorithmForwardConcretization
        lin_fun_extractor_algorithm = fwd_bwd_alg(
            fwd_bound_applier, extracting_concretizer
        )
        # We get one target_linfuns per output.
        target_linfuns, _ = bound_propagation.bound_propagation(
            lin_fun_extractor_algorithm,
            f_wrapped,
            bound,
            graph_simplifier=default_simplifier_without_linear,
        )

        return CROWNResult(
            lC=target_linfuns[0].lin_coeffs[:out_len, :],
            uC=-target_linfuns[0].lin_coeffs[out_len:, :],
            ld=target_linfuns[0].offset[:out_len],
            ud=-target_linfuns[0].offset[out_len:],
        )
        # return target_linfuns

    return F
