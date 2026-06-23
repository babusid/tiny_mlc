"""Evaluator for TMLC computational graphs."""

from tmlc.ndarray import ndarray
from tmlc.tensor import Tensor
from tmlc.graph.graph import Graph

def run(inputs: dict[Tensor, ndarray], graph: Graph ) -> list[list[ndarray]]:
    outputs = graph.outputs
    topo_sort = graph.topo_sort
    intermediates: dict[Tensor, list[ndarray]] = {}
    for node in topo_sort:
        if node in inputs:
            intermediates[node] = [inputs[node]]
        else:
            input_values = [intermediates[input][0] for input in node.inputs]
            assert len(input_values) == len(node.inputs), (
                "Mismatch in number of input values and node inputs"
            )
            intermediates[node] = node.op.compute(input_values)
    output: list[list[ndarray]] = []
    for out in outputs:
        output.append(intermediates[out])

    return output



def compile() -> None:
    return


def run_compiled() -> None:
    return
