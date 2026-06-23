from collections import defaultdict
from tmlc.tensor import Tensor
from tmlc.ops.ops_basic import Constant, Input
from tmlc.ops.ops_shape import ones_like
from tmlc.graph.graph import Graph

def differentiate(graph: Graph, output_node: Tensor, target_nodes: list[Tensor]) -> Graph:

    visited: set[Tensor] = set()
    rev_topo_sort: list[Tensor] = []
    rev_topo_sort = [t for t in graph.topo_sort]
    rev_topo_sort.reverse()

    output_grad = ones_like(output_node, "output_grad")

    # track which nodes in the graph we actually have to
    # compute gradients for. This includes the targets,
    # the output gradient, and everything on the path between them
    target_set = set(target_nodes + [output_node])
    visited = set()

    def generate_target_set(tensor: Tensor) -> bool:
        # if we've already explored this node, just return whether it ended up a target
        if tensor in visited:
            return tensor in target_set
        visited.add(tensor)

        # inputs and constants are leaves: they're only targets if explicitly requested
        if isinstance(tensor.op, Input) or isinstance(tensor.op, Constant):
            return tensor in target_set

        # always recurse, even if `tensor` is already an explicit target, since ancestors
        # further up the graph still need to be connected through to it
        reached_target = tensor in target_set
        for input in tensor.inputs:
            if generate_target_set(input):
                reached_target = True

        if reached_target:
            target_set.add(tensor)
        return reached_target

    _ = generate_target_set(output_node)

    # now we have all nodes we have to compute gradients for in target_set
    # for each node, we have to track what is coming in backwards
    node_grad_incoming: dict[Tensor, list[Tensor]] = defaultdict(list)
    # output node just gets the all one output gradient
    node_grad_incoming[output_node] = [output_grad]
    # map tensor to the aggregate of its input gradients
    node_grad: dict[Tensor, Tensor] = {}
    for node in rev_topo_sort:
        if node not in target_set:
            continue
        # get all incoming partial gradients, and aggregate them
        incoming_grad = node_grad_incoming[node]
        sum_grad = incoming_grad[0]
        for grad in incoming_grad[1:]:
            sum_grad += grad
        node_grad[node] = sum_grad
        # pass in aggregate gradient and calculate the gradients
        # wrt this nodes inputs
        input_grads = node.op.gradients(node, sum_grad)
        for input, input_grad in zip(node.inputs, input_grads):
            # pass in the appropriate gradients
            node_grad_incoming[input].append(input_grad)

    bwd_outputs = [node_grad[target] for target in target_nodes]
    return Graph(graph.inputs, bwd_outputs)


