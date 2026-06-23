"""
Explicit graph object that traces a computation graph built from Tensor objects.
Must be built in order to actually run the computation graph, as well as to apply
graph optimizations for increased performance. Building a Graph is the first part
of the compilation process.
"""

from typing import Callable
import copy
from functools import reduce
from tmlc.tensor import Tensor
from tmlc.graph.util.topo_sort import dfs_helper_topo_sort


class Graph:
    inputs: list[Tensor]
    outputs: list[Tensor]
    topo_sort: list[Tensor]

    def __init__(self, inputs: list[Tensor], outputs: list[Tensor]) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.topo_sort = self._build_topo_sort()

    def _build_topo_sort(self) -> list[Tensor]:
        """Return a topological sort of the graph's nodes."""
        visited: set[Tensor] = set()
        _topo: list[Tensor] = []
        for node in self.outputs:
            dfs_helper_topo_sort(node, visited, _topo)
        return _topo

    def apply_transforms(self, transform_fns: list[Callable[["Graph"], "Graph"]]) -> "Graph":
        """Apply a transform pipeline to the graph"""
        # TODO: typehint for the callable might be too restrictive / wrong? need to think more
        init: "Graph" = copy.deepcopy(self)
        return reduce(lambda graph, fn: fn(graph), transform_fns, init)

