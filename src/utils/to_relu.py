#!/usr/bin/env python3
# Replace all Sigmoid ops in an ONNX model with Relu and save a new model.
# Works recursively on If/Loop/Scan subgraphs. Validates the result.

import argparse
from typing import Tuple
import onnx
from onnx import ModelProto, GraphProto, checker


def _replace_in_graph(graph: GraphProto) -> int:
    """Recursively replace Sigmoid -> Relu. Returns number of replacements."""
    replaced = 0
    for node in graph.node:
        if node.op_type == "Sigmoid":
            node.op_type = "Relu"
            node.domain = ""   # standard ONNX ops live in empty domain
            replaced += 1

        # Recurse into graph-valued attributes (If/Loop/Scan, etc.)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                replaced += _replace_in_graph(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sg in attr.graphs:
                    replaced += _replace_in_graph(sg)
    return replaced


def _count_ops(graph: GraphProto, op_type: str) -> int:
    total = sum(1 for n in graph.node if n.op_type == op_type)
    for n in graph.node:
        for attr in n.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                total += _count_ops(attr.g, op_type)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sg in attr.graphs:
                    total += _count_ops(sg, op_type)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to source .onnx model")
    ap.add_argument("-o", "--output", default=None,
                    help="Path to write the modified .onnx (default: add _relu suffix)")
    args = ap.parse_args()

    src = args.input
    dst = args.output or (src.rsplit(".onnx", 1)[0] + "_relu.onnx")

    model: ModelProto = onnx.load(src)

    sig_before = _count_ops(model.graph, "Sigmoid")
    relu_before = _count_ops(model.graph, "Relu")
    print(f"Sigmoid BEFORE: {sig_before} | Relu BEFORE: {relu_before}")

    num_replaced = _replace_in_graph(model.graph)

    # Update metadata (optional)
    model.producer_name = (model.producer_name or "onnx") + "+sigmoid2relu"
    # Save and check
    onnx.save(model, dst)
    onnx_model_new = onnx.load(dst)
    checker.check_model(onnx_model_new)

    sig_after = _count_ops(onnx_model_new.graph, "Sigmoid")
    relu_after = _count_ops(onnx_model_new.graph, "Relu")

    print(f"Replacements performed: {num_replaced}")
    print(f"Sigmoid AFTER: {sig_after} | Relu AFTER: {relu_after}")
    print(f"Saved modified model to: {dst}")


if __name__ == "__main__":
    main()
