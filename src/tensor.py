from typing import Any
from numpy._core.numeric import ndarray
from typing_extensions import override
from abc import ABC, abstractmethod
import numpy as np


class Tensor:
    """
    A Tensor is a node in a computational graph, representing a multi-dimensional array.
    Tensors are the inputs to tensor operations, which output new tensors. A sequence of Tensor Operations
    chained together produces a computational graph, which we can compile and optimize.
    """

    # input tensors that feed this one (parent nodes in the graph)
    input: list["Tensor"]

    # the computation done on this tensor to emit new ones
    op: "TensorOp"

    # Tensor label
    label: str

    # TODO: does tensor need to store shape / dtype? does it need to store a value buffer?
    # rn thinking no - tensor should just be node graph. array should be separate data structure
    # that is passed through the evaluator.
    # BUT, in eager mode might be good to associate buffers with the tensors for debugging
    # / visualization purposes.

    def __init__(
        self, inputs: list["Tensor"], op: "TensorOp", label: str | None = None
    ):
        self.input = inputs
        self.op = op
        if label is None:
            self.label = self.op.__class__.__name__
        else:
            self.label = label

    @override
    def __str__(self):
        return f"Tensor(op={self.label}, inputs={[str(i) for i in self.input]})"

    @override
    def __repr__(self):
        return self.__str__()


class TensorOp(ABC):
    """
    TensorOp interface represents an operation that can be performed on Tensors. 
    """
    @abstractmethod
    def __call__(
        self, inputs: list[Tensor] | None = None, label: str | None = None
    ) -> Tensor:
        """
        When a TensorOp is called, it should create a new Tensor that represents the output of this operation.
        """
        raise NotImplementedError("TensorOp subclasses must implement __call__")

    @abstractmethod
    def compute(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        """
        Given the input arrays, compute the output arrays of this operation. This is used by the
        evaluator to compute the values of the output tensors in the graph.
        This operates on concrete arrays to actually determine a concrete value, and is used for eager
        mode evaluation.
        """
        raise NotImplementedError("TensorOp subclasses must implement compute()")

    @abstractmethod
    def gradients(self, inputs: list[Tensor]) -> list[Tensor]:
        """
        Given the input tensors, compute the gradients of this operation with respect to its inputs.
        This is used by the autograd engine to compute gradients for backpropagation.
        This operates on Tensors instead of concrete values because gradients are represented as just
        more tensors in the graph, evaluated only when necessary.
        """
        raise NotImplementedError("TensorOp subclasses must implement gradients()")

    @abstractmethod
    def emit_ir(self, inputs: list[str]) -> str:
        # TODO: may need to update function signature here. Do we need input tensor labels?
        """
        If compiling the graph, each TensorOp needs to emit IR that represents this operations computation.
        The compiler composes the graphs full IR to optimize and generate the final code.
        """
        raise NotImplementedError("TensorOp subclasses must implement emit_ir()")


class Input(TensorOp):
    """
    Input denotes an input to the computational graph. It cannot accept any input Tensors,
    run compute or gradients, and is just a leaf node placeholder.
    """

    @override
    def __call__(
        self, inputs: list[Tensor] | None = None, label: str | None = None
    ) -> Tensor:
        return Tensor(inputs=[], op=self, label=label)

    @override
    def compute(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        raise RuntimeError(
            "Input op does not have a compute implementation.",
            "Did you forget to assign an input node a value before evaluating the graph?",
        )

    @override
    def gradients(self, inputs: list[Tensor]) -> list[Tensor]:
        raise RuntimeError("Input Ops don't have gradients")

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        # TODO: Figure out what to do here not sure what emitting ir for a leaf node looks like
        return ""

class Add(TensorOp):
    pass

class AddConst(TensorOp):
    pass

class Mul(TensorOp):
    pass

class MulConst(TensorOp):
    pass

class Div(TensorOp):
    pass

class DivConst(TensorOp):
    pass

class Matmul(TensorOp):
    pass

class ZerosLike(TensorOp):
    pass

class OnesLike(TensorOp):
    pass

class Reshape(TensorOp):
    pass

class BroadcastTo(TensorOp):
    pass


# Singleton factory instances of tensor operations. 
_input = Input()

# Functional wrappers on singletons
# Note: singleton pattern allows us to have a generic interface for all TensorOps, 
# while adding op-specific usage gates at the wrapper level. For example, we can enforce
# that the input op doesn't accept any input tensors.

def input(label: str | None = None) -> Tensor:
    """
    Create an input Tensor with an optional label. 
    This is used to denote the inputs to the computational graph.
    All input nodes must be assigned a value at evaluation time. 
    """
    return _input(label=label)


