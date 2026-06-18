from typing_extensions import override
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray


class Tensor:
    """A Tensor is a node in a computational graph, representing a multi-dimensional array.
    Tensors are the inputs to tensor operations, which output new tensors. A sequence of Tensor
    Operations chained together produces a computational graph, which we can compile and optimize.
    Tensors do NOT actually hold data themseleves, but rather represent the flow of data through
    the graph. The actual data is held in buffers that are supplied at evaluation time.
    """

    # input tensors that feed this one (parent nodes in the graph)
    inputs: list["Tensor"]

    # the computation done on this tensor to emit new ones
    op: "TensorOp"

    # Tensor label
    label: str

    # Tensor shape
    # Can be inferred from the inputs and the operation
    # or can be set explicitly for leaf nodes (input, constant)
    shape: tuple[int, ...]

    # Tensor dtype
    dtype: str

    def __init__(
        self,
        inputs: list["Tensor"],
        op: "TensorOp",
        shape: tuple[int, ...],
        label: str | None = None,
        dtype: str = "float32",
    ):
        self.inputs = inputs
        self.op = op
        if label is None:
            self.label = self.op.__class__.__name__
        else:
            self.label = label

        self.shape = shape

        # TODO: support more dtypes
        # TODO: inheirit dtypes from input tensors
        # TODO: dtype promotion logic for mismatched input tensor
        self.dtype = "float32"

    @override
    def __str__(self):
        return f"Tensor(op={self.label}, inputs={[str(i) for i in self.inputs]})"

    @override
    def __repr__(self):
        return self.__str__()

    def __add__(self, other: "Tensor|float|int") -> "Tensor":
        if isinstance(other, (int, float)):
            other = constant(other, label=str(other))
        return add(self, other)

    def __radd__(self, other: "Tensor|float|int") -> "Tensor":
        if isinstance(other, (int, float)):
            other = constant(other, label=str(other))
        return add(self, other)

    def __mul__(self, other: "Tensor|float|int") -> "Tensor":
        if isinstance(other, (int, float)):
            other = constant(other, label=str(other))
        return mul(self, other)

    def __rmul__(self, other: "Tensor|float|int") -> "Tensor":
        if isinstance(other, (int, float)):
            other = constant(other, label=str(other))
        return mul(self, other)

    def __truediv__(self, other: "Tensor|float|int") -> "Tensor":
        if isinstance(other, (int, float)):
            other = constant(1 / other, label=str(1 / other))
        return div(self, other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return mm(self, other)

    @property
    def T(self) -> "Tensor":
        return transpose(self)


class ConstantTensor(Tensor):
    """
    Constant is a special type of Tensor that holds a buffer that is managed outside of the graph.
    It is operationally equivalent to an Input node, except that its value is determined at creation
    time, rather than supplied at evaluation time.
    It is a leaf node in the graph and does not have any input tensors.
    The value of a Constant tensor is stored in the `constval` field.
    """

    constval: ndarray

    def __init__(self, value: ndarray, op: "TensorOp", label: str | None = None):
        super().__init__(inputs=[], op=op, label=label, shape=value.shape)
        self.constval = value


class TensorOp(ABC):
    """TensorOp interface represents an operation that can be performed on Tensors."""

    @abstractmethod
    def __call__(
        self,
        inputs: list[Tensor],
        label: str | None = None,
    ) -> Tensor:
        """When a TensorOp is called, it should create a new Tensor that represents the output of
        this operation."""
        raise NotImplementedError("TensorOp subclasses must implement __call__")

    @abstractmethod
    def infer_shape(self, inputs: list[Tensor]) -> tuple[int, ...]:
        """
        Given the input tensors (which contain their shapes), infer the shape of the
        output tensor that this operation will produce.
        """
        raise NotImplementedError("TensorOp subclasses must implement infer_shape()")

    @abstractmethod
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
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


class Constant(TensorOp):
    value: ndarray

    def __init__(self, value: float | int | ndarray):
        if isinstance(value, (float, int)):
            value = np.array(value)
        self.value = value

    @override
    def __call__(
        self,
        inputs: list[Tensor],
        label: str | None = None,
    ) -> Tensor:
        assert inputs is None or len(inputs) == 0, "Constant op cannot accept any input tensors"
        return ConstantTensor(value=self.value, op=self, label=label)

    @override
    def infer_shape(self, inputs: list[Tensor]) -> tuple[int, ...]:
        assert inputs is None or len(inputs) == 0, "Constant op cannot accept any input tensors"
        return self.value.shape

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert inputs is None or len(inputs) == 0, "Constant op cannot accept any input tensors"
        return [self.value]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return []  # Constant nodes do not have gradients

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        # TODO: Figure out what to do here not sure what emitting ir for a leaf node looks like
        return ""


class Input(TensorOp):
    """Input denotes an input to the computational graph.

    It cannot accept any Tensors, run compute or gradients, and is just a leaf node placeholder.
    """

    shape: tuple[int, ...]

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    @override
    def __call__(
        self,
        inputs: list[Tensor],
        label: str | None = None,
    ) -> Tensor:
        assert inputs is None or len(inputs) == 0, "Input op cannot accept any input tensors"
        return Tensor(
            inputs=[],
            op=self,
            shape=self.infer_shape(inputs=inputs),
            label=label,
        )

    @override
    def infer_shape(self, inputs: list[Tensor]) -> tuple[int, ...]:
        assert inputs is None or len(inputs) == 0, "Input op cannot accept any input tensors"
        return self.shape

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        raise RuntimeError(
            "Input op does not have a compute implementation.",
            "Did you forget to assign an input node a value before evaluating the graph?",
        )

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return []  # Input nodes do not have gradients

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        # TODO: Figure out what to do here not sure what emitting ir for a leaf node looks like
        return ""


class Add(TensorOp):
    @override
    def __call__(
        self,
        inputs: list[Tensor],
        label: str | None = None,
    ) -> Tensor:
        return Tensor(
            inputs=inputs,
            op=self,
            shape=self.infer_shape(inputs=inputs),
            label=label,
        )

    @override
    def infer_shape(self, inputs: list[Tensor]) -> tuple[int, ...]:
        # we force broadcasting to happen as a separate graph node (no implicit broadcast)
        # so we just return the shape of the first input tensor here
        assert len(inputs) == 2, "Add op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, "Add op requires tensors to have the same shape"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 2, "Add op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, "Add op requires tensors to have the same shape"
        return [inputs[0] + inputs[1]]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        # dL/dx = dL/dy * dy/dx = incoming_grad * 1
        return [incoming_grad, incoming_grad]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Mul(TensorOp):
    @override
    def __call__(
        self,
        inputs: list[Tensor],
        label: str | None = None,
    ) -> Tensor:
        return Tensor(
            inputs=inputs,
            op=self,
            shape=self.infer_shape(inputs=inputs),
            label=label,
        )

    @override
    def infer_shape(self, inputs: list[Tensor]) -> tuple[int, ...]:
        assert len(inputs) == 2, "Mul op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, "Mul op requires tensors to have the same shape"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 2, "Mul op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, "Mul op requires tensors to have the same shape"
        return [inputs[0] * inputs[1]]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        # dL/dx = dL/dy * dy/dx = incoming_grad * y
        return [tensor.inputs[1] * incoming_grad, tensor.inputs[0] * incoming_grad]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Div(TensorOp):
    @override
    def __call__(
        self,
        inputs: list[Tensor],
        label: str | None = None,
    ) -> Tensor:
        return Tensor(
            inputs=inputs,
            op=self,
            shape=self.infer_shape(inputs=inputs),
            label=label,
        )

    @override
    def infer_shape(self, inputs: list[Tensor]) -> tuple[int, ...]:
        assert len(inputs) == 2, "Div op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, "Div op requires tensors to have the same shape"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 2, "Div op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, "Div op requires tensors to have the same shape"
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


class Transpose(TensorOp):
    @override
    def __call__(
        self,
        inputs: list[Tensor],
        label: str | None = None,
    ) -> Tensor:
        return Tensor(
            inputs=inputs,
            op=self,
            shape=self.infer_shape(inputs=inputs),
            label=label,
        )

    @override
    def infer_shape(self, inputs: list[Tensor]) -> tuple[int, ...]:
        assert len(inputs) == 1, "Transpose op requires exactly 1 input tensor"
        assert len(inputs[0].shape) == 2, "Transpose op requires a 2D input tensor"
        return (inputs[0].shape[1], inputs[0].shape[0])

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 1, "Transpose op requires exactly 1 input tensor"
        assert inputs[0].ndim == 2, "Transpose op requires a 2D input tensor"
        return [inputs[0].T]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [incoming_grad.T]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Matmul(TensorOp):
    @override
    def __call__(
        self,
        inputs: list[Tensor],
        label: str | None = None,
    ) -> Tensor:
        return Tensor(
            inputs=inputs,
            op=self,
            shape=self.infer_shape(inputs=inputs),
            label=label,
        )

    @override
    def infer_shape(self, inputs: list[Tensor]) -> tuple[int, ...]:
        assert len(inputs) == 2, "Matmul op requires exactly 2 input tensors"
        assert len(inputs[0].shape) == 2 and len(inputs[1].shape) == 2, (
            "Matmul op requires 2D input tensors"
        )
        assert inputs[0].shape[1] == inputs[1].shape[0], "Matmul input shapes are incompatible"
        return (inputs[0].shape[0], inputs[1].shape[1])

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 2, "Matmul op requires exactly 2 input tensors"
        assert inputs[0].ndim == 2 and inputs[1].ndim == 2, "Matmul op requires 2D input tensors"
        return [inputs[0] @ inputs[1]]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        # dL/dA = dL/dC * dC/dA = incoming_grad * B^T
        # dL/dB = dL/dC * dC/dB = A^T * incoming_grad
        return [incoming_grad @ tensor.inputs[1].T, tensor.inputs[0].T @ incoming_grad]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class ZerosLike(TensorOp):
    @override
    def __call__(
        self,
        inputs: list[Tensor],
        label: str | None = None,
    ) -> Tensor:
        pass

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        pass

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        pass

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class OnesLike(TensorOp):
    @override
    def __call__(
        self,
        inputs: list[Tensor],
        label: str | None = None,
    ) -> Tensor:
        pass

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        pass

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        pass

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Reshape(TensorOp):
    @override
    def __call__(
        self,
        inputs: list[Tensor],
        label: str | None = None,
    ) -> Tensor:
        pass

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        pass

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        pass

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class BroadcastTo(TensorOp):
    @override
    def __call__(
        self,
        inputs: list[Tensor],
        label: str | None = None,
    ) -> Tensor:
        pass

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        pass

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        pass

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


def constant(value: float | int | ndarray, label: str | None = None) -> Tensor:
    """Create a constant Tensor with the given value and an optional label.
    This is used to denote constant values in the computational graph that are not supplied at
    evaluation time, but rather determined at graph construction time.
    """
    return Constant(value)(inputs=[], label=label)


def input(shape: tuple[int, ...], label: str | None = None) -> Tensor:
    """Create an input Tensor with an optional label.
    This is used to denote the inputs to the computational graph.
    All input nodes must be assigned a
    value at evaluation time.
    """
    return Input(shape=shape)(inputs=[], label=label)


# TODO: add broadcasting for ops that need it in the wrapping helper funcs


def add(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    """Add two tensors elementwise, returning a new tensor that represents the output of this
    operation."""
    return Add()(inputs=[t1, t2], label=label)


def mul(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    """Multiply two tensors elementwise, returning a new tensor that represents the output of this
    operation."""
    return Mul()(inputs=[t1, t2], label=label)


def div(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    """Divide two tensors elementwise, returning a new tensor that represents the output of this
    operation."""
    return Div()(inputs=[t1, t2], label=label)


def mm(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    """Matrix multiply two tensors, returning a new tensor that represents the output of this
    operation."""
    return Matmul()(inputs=[t1, t2], label=label)


def transpose(t: Tensor, label: str | None = None) -> Tensor:
    """Transpose a tensor, returning a new tensor that represents the output of this operation."""
    return Transpose()(inputs=[t], label=label)
