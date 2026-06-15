from typing_extensions import override
from abc import ABC, abstractmethod
import numpy as np


class Tensor:
    """A Tensor is a node in a computational graph, representing a multi-dimensional array.

    Tensors are the inputs to tensor operations, which output new tensors. A sequence of Tensor
    Operations chained together produces a computational graph, which we can compile and optimize.
    """

    # input tensors that feed this one (parent nodes in the graph)
    inputs: list["Tensor"]

    # the computation done on this tensor to emit new ones
    op: "TensorOp"

    # Tensor label
    label: str

    # constant value field for tensors associated with constant ops
    constval: float | None

    def __init__(
        self,
        inputs: list["Tensor"],
        op: "TensorOp",
        label: str | None = None,
        constval: float | None = None,
    ):
        self.inputs = inputs
        self.op = op
        self.constval = constval
        if label is None:
            self.label = self.op.__class__.__name__
        else:
            self.label = label

    @override
    def __str__(self):
        return f"Tensor(op={self.label}, inputs={[str(i) for i in self.inputs]})"

    @override
    def __repr__(self):
        return self.__str__()

    def __add__(self, other: "Tensor|float|int") -> "Tensor":
        if isinstance(other, (int, float)):
            return add_const(self, other)
        return add(self, other)

    def __radd__(self, other: "Tensor|float|int") -> "Tensor":
        if isinstance(other, (int, float)):
            return add_const(self, other)
        return add(self, other)

    def __mul__(self, other: "Tensor|float|int") -> "Tensor":
        if isinstance(other, (int, float)):
            return mul_const(self, other)
        return mul(self, other)

    def __rmul__(self, other: "Tensor|float|int") -> "Tensor":
        if isinstance(other, (int, float)):
            return mul_const(self, other)
        return mul(self, other)

    def __truediv__(self, other: "Tensor|float|int") -> "Tensor":
        if isinstance(other, (int, float)):
            return mul_const(self, 1 / other)
        return div(self, other)


class TensorOp(ABC):
    """TensorOp interface represents an operation that can be performed on Tensors."""

    @abstractmethod
    def __call__(
        self,
        inputs: list[Tensor],
        const_inputs: list[float | int] | None = None,
        label: str | None = None,
    ) -> Tensor:
        """When a TensorOp is called, it should create a new Tensor that represents the output of
        this operation."""
        raise NotImplementedError("TensorOp subclasses must implement __call__")

    @abstractmethod
    def compute(
        self, inputs: list[np.ndarray], const_inputs: list[float | int] | None = None
    ) -> list[np.ndarray]:
        """Given the input arrays, compute the output arrays of this operation.

        This is used by the evaluator to compute the values of the output tensors in the graph. This
        operates on concrete arrays to actually determine a concrete value, and is used for eager
        mode evaluation.
        """
        raise NotImplementedError("TensorOp subclasses must implement compute()")

    @abstractmethod
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        """Given the output of the forward `call` method and the incoming gradient from the
        backwards pass, this method calculates the gradients to propagate to the inputs.

        The calculated gradients must be arranged in a list that corresponds to the original
        ordering of the input tensors.
        """
        raise NotImplementedError("TensorOp subclasses must implement gradients()")

    @abstractmethod
    def emit_ir(self, inputs: list[str]) -> str:
        # TODO: may need to update function signature here. Do we need input tensor labels?
        """If compiling the graph, each TensorOp needs to emit IR that represents this operations
        computation.

        The compiler composes the graphs full IR to optimize and generate the final code.
        """
        raise NotImplementedError("TensorOp subclasses must implement emit_ir()")


class Input(TensorOp):
    """Input denotes an input to the computational graph.

    It cannot accept any Tensors, run compute or gradients, and is just a leaf node placeholder.
    """

    @override
    def __call__(
        self,
        inputs: list[Tensor],
        const_inputs: list[float] | None = None,
        label: str | None = None,
    ) -> Tensor:
        assert inputs is None or len(inputs) == 0, "Input op cannot accept any input tensors"
        assert const_inputs is None or len(const_inputs) == 0, (
            "Input op cannot accept any constant inputs"
        )
        return Tensor(inputs=[], op=self, label=label)

    @override
    def compute(
        self,
        inputs: list[np.ndarray],
        const_inputs: list[float] | None = None,
    ) -> list[np.ndarray]:
        raise RuntimeError(
            "Input op does not have a compute implementation.",
            "Did you forget to assign an input node a value before evaluating the graph?",
        )

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        raise RuntimeError("Input Ops don't have gradients")

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        # TODO: Figure out what to do here not sure what emitting ir for a leaf node looks like
        return ""


class Add(TensorOp):
    @override
    def __call__(
        self,
        inputs: list[Tensor],
        const_inputs: list[float] | None = None,
        label: str | None = None,
    ) -> Tensor:
        assert len(inputs) == 2, "Add op requires exactly 2 input tensors"
        assert const_inputs is None or len(const_inputs) == 0, (
            "Add op does not accept constant inputs"
        )
        return Tensor(inputs=inputs, op=self, label=label)

    @override
    def compute(
        self, inputs: list[np.ndarray], const_inputs: list[float] | None = None
    ) -> list[np.ndarray]:
        assert len(inputs) == 2, "Add op requires exactly 2 input tensors"
        assert const_inputs is None or len(const_inputs) == 0, (
            "Add op does not accept constant inputs"
        )

        return [inputs[0] + inputs[1]]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        # dL/dx = dL/dy * dy/dx = incoming_grad * 1
        return [incoming_grad, incoming_grad]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class AddConst(TensorOp):
    @override
    def __call__(
        self,
        inputs: list[Tensor],
        const_inputs: list[float | int] | None = None,
        label: str | None = None,
    ) -> Tensor:
        assert len(inputs) == 1, "AddConst op requires exactly 1 input tensors"
        assert const_inputs and len(const_inputs) == 1, (
            "AddConst op requires exactly 1 constant input"
        )
        return Tensor(inputs=inputs, op=self, label=label, constval=const_inputs[0])

    @override
    def compute(
        self, inputs: list[np.ndarray], const_inputs: list[float | int] | None = None
    ) -> list[np.ndarray]:
        assert len(inputs) == 1, "AddConst op requires exactly 1 input tensors"
        assert const_inputs and len(const_inputs) == 1, (
            "AddConst op requires exactly 1 constant input"
        )
        return [inputs[0] + const_inputs[0]]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [incoming_grad]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Mul(TensorOp):
    @override
    def __call__(
        self,
        inputs: list[Tensor],
        const_inputs: list[float] | None = None,
        label: str | None = None,
    ) -> Tensor:
        assert len(inputs) == 2, "Mul op requires exactly 2 input tensors"
        assert const_inputs is None or len(const_inputs) == 0, (
            "Mul op does not accept constant inputs"
        )
        return Tensor(inputs=inputs, op=self, label=label)

    @override
    def compute(
        self, inputs: list[np.ndarray], const_inputs: list[float] | None = None
    ) -> list[np.ndarray]:
        assert len(inputs) == 2, "Mul op requires exactly 2 input tensors"
        assert const_inputs is None or len(const_inputs) == 0, (
            "Mul op does not accept constant inputs"
        )
        return [inputs[0] * inputs[1]]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        # dL/dx = dL/dy * dy/dx = incoming_grad * y
        return [tensor.inputs[1] * incoming_grad, tensor.inputs[0] * incoming_grad]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class MulConst(TensorOp):
    @override
    def __call__(
        self,
        inputs: list[Tensor],
        const_inputs: list[float | int] | None = None,
        label: str | None = None,
    ) -> Tensor:
        assert len(inputs) == 1, "MulConst op requires exactly 1 input tensors"
        assert const_inputs and len(const_inputs) == 1, (
            "MulConst op requires exactly 1 constant input",
        )

        return Tensor(inputs=inputs, op=self, label=label, constval=const_inputs[0])

    @override
    def compute(
        self, inputs: list[np.ndarray], const_inputs: list[float] | None = None
    ) -> list[np.ndarray]:
        assert len(inputs) == 1, "Mul op requires exactly 1 input array"
        assert const_inputs and len(const_inputs) == 1, (
            "MulConst op requires exactly 1 constant input"
        )

        return [inputs[0] * const_inputs[0]]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        assert tensor.constval is not None, "MulConst op requires a constant value"

        # dL/dx = dL/dy * dy/dx = incoming_grad * y
        return [tensor.constval * incoming_grad]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Div(TensorOp):
    @override
    def __call__(
        self,
        inputs: list[Tensor],
        const_inputs: list[float | int] | None = None,
        label: str | None = None,
    ) -> Tensor:
        assert len(inputs) == 2, "Div op requires exactly 2 input tensors"
        assert const_inputs is None or len(const_inputs) == 0, (
            "Div op does not accept constant inputs"
        )

        return Tensor(inputs=inputs, op=self, label=label)

    @override
    def compute(
        self, inputs: list[np.ndarray], const_inputs: list[float | int] | None = None
    ) -> list[np.ndarray]:
        assert len(inputs) == 2, "Div op requires exactly 2 input tensors"
        assert const_inputs is None or len(const_inputs) == 0, (
            "Div op does not accept constant inputs",
        )

        return [inputs[0] / inputs[1]]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        # dy/da = dy/dc * dc/da = dy/dc * dda(a*b^-1) = dy/dc * 1/b
        # dy/db = dy/dc * dc/db = dy/dc * ddb(a*b^-1) = dy/dc * -ab^-2
        return [
            incoming_grad / tensor.inputs[1],
            (incoming_grad * -1 * tensor.inputs[0]) / (tensor.inputs[1] * tensor.inputs[1]),
        ]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


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
_add = Add()
_addc = AddConst()
_mul = Mul()
_mulc = MulConst()
_div = Div()
_mm = Matmul()
_zlike = ZerosLike()
_olike = OnesLike()
_reshape = Reshape()
_brodacast = BroadcastTo()

# Functional wrappers on singletons
# Note: singleton pattern allows us to have a generic interface for all TensorOps,
# while adding op-specific usage gates at the wrapper level. For example, we can enforce
# that the input op doesn't accept any input tensors.


def input(label: str | None = None) -> Tensor:
    """Create an input Tensor with an optional label.
    This is used to denote the inputs to the computational graph.
    All input nodes must be assigned a
    value at evaluation time.
    """
    return _input(inputs=[], label=label)


def add(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    """Add two tensors elementwise, returning a new tensor that represents the output of this
    operation."""
    return _add(inputs=[t1, t2], label=label)


def mul(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    """Multiply two tensors elementwise, returning a new tensor that represents the output of this
    operation."""
    return _mul(inputs=[t1, t2], label=label)


def mul_const(t: Tensor, c: float | int, label: str | None = None) -> Tensor:
    """Multiply a tensor by a constant, returning a new tensor that represents the output of this
    operation."""
    return _mulc(inputs=[t], const_inputs=[c], label=label)


def div(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    """Divide two tensors elementwise, returning a new tensor that represents the output of this
    operation."""
    return _div(inputs=[t1, t2], label=label)
