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

    def __sub__(self, other: "Tensor|float|int") -> "Tensor":
        if isinstance(other, (int, float)):
            other = constant(other, label=str(other))
        return add(self, negate(other))

    def __rsub__(self, other: "Tensor|float|int") -> "Tensor":
        if isinstance(other, (int, float)):
            other = constant(other, label=str(other))
        return add(other, negate(self))

    def __neg__(self) -> "Tensor":
        return negate(self)

    def __pow__(self, other: "Tensor|float|int") -> "Tensor":
        if isinstance(other, (int, float)):
            other = constant(other, label=str(other))
        return power(self, other)

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
    axes: tuple[int, int] | None

    def __init__(self, axes: tuple[int, int] | None = None):
        self.axes = axes

    def _permutation(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        assert len(shape) >= 2, "Transpose op requires at least 2 dimensions"
        axes = self.axes if self.axes is not None else (-2, -1)
        a1, a2 = axes
        ndim = len(shape)
        if a1 < 0:
            a1 += ndim
        if a2 < 0:
            a2 += ndim
        assert 0 <= a1 < ndim and 0 <= a2 < ndim, "Transpose axes are out of bounds"

        permutation = list(range(ndim))
        permutation[a1], permutation[a2] = permutation[a2], permutation[a1]
        return tuple(permutation)

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
        permutation = self._permutation(shape=inputs[0].shape)
        return tuple(inputs[0].shape[axis] for axis in permutation)

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 1, "Transpose op requires exactly 1 input tensor"
        return [np.transpose(inputs[0], axes=self._permutation(shape=inputs[0].shape))]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [transpose(incoming_grad, axes=self.axes)]

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


def _normalize_axes(
    axes: tuple[int, ...] | int | None, shape: tuple[int, ...]
) -> tuple[int, ...] | None:
    if axes is None:
        return None
    if isinstance(axes, int):
        axes = (axes,)

    normalized:list[int] = []
    for axis in axes:
        if axis < 0:
            axis += len(shape)
        assert 0 <= axis < len(shape), "Axis is out of bounds"
        normalized.append(axis)
    assert len(set(normalized)) == len(normalized), "Axes must be unique"
    return tuple(sorted(normalized))


class Summation(TensorOp):
    axes: tuple[int, ...] | None

    def __init__(self, axes: tuple[int, ...] | int | None = None):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

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
        assert len(inputs) == 1, "Summation op requires exactly 1 input tensor"
        axes = _normalize_axes(axes=self.axes, shape=inputs[0].shape)
        if axes is None:
            return ()
        return tuple(dim for axis, dim in enumerate(inputs[0].shape) if axis not in axes)

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 1, "Summation op requires exactly 1 input tensor"
        return [np.sum(inputs[0], axis=self.axes)]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        input_shape = tensor.inputs[0].shape
        axes = _normalize_axes(axes=self.axes, shape=input_shape)
        if axes is None:
            reshaped_grad = reshape(incoming_grad, shape=(1,) * len(input_shape))
        else:
            grad_shape = tuple(1 if axis in axes else dim for axis, dim in enumerate(input_shape))
            reshaped_grad = reshape(incoming_grad, shape=grad_shape)
        return [broadcast_to(reshaped_grad, shape=input_shape)]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Negate(TensorOp):
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
        assert len(inputs) == 1, "Negate op requires exactly 1 input tensor"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 1, "Negate op requires exactly 1 input tensor"
        return [-inputs[0]]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [negate(incoming_grad)]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Exp(TensorOp):
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
        assert len(inputs) == 1, "Exp op requires exactly 1 input tensor"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 1, "Exp op requires exactly 1 input tensor"
        return [np.exp(inputs[0])]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [incoming_grad * tensor]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Log(TensorOp):
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
        assert len(inputs) == 1, "Log op requires exactly 1 input tensor"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 1, "Log op requires exactly 1 input tensor"
        return [np.log(inputs[0])]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [incoming_grad / tensor.inputs[0]]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Pow(TensorOp):
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
        assert len(inputs) == 2, "Power op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, (
            "Power op requires tensors to have the same shape"
        )
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 2, "Power op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, (
            "Power op requires tensors to have the same shape"
        )
        return [inputs[0] ** inputs[1]]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        lhs, rhs = tensor.inputs
        return [
            incoming_grad * rhs * power(lhs, rhs - 1),
            incoming_grad * tensor * log(lhs),
        ]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Tanh(TensorOp):
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
        assert len(inputs) == 1, "Tanh op requires exactly 1 input tensor"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 1, "Tanh op requires exactly 1 input tensor"
        return [np.tanh(inputs[0])]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [incoming_grad * (1 - tensor * tensor)]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class LogSumExp(TensorOp):
    axes: tuple[int, ...] | None

    def __init__(self, axes: tuple[int, ...] | int | None = None):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

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
        assert len(inputs) == 1, "LogSumExp op requires exactly 1 input tensor"
        axes = _normalize_axes(axes=self.axes, shape=inputs[0].shape)
        if axes is None:
            return ()
        return tuple(dim for axis, dim in enumerate(inputs[0].shape) if axis not in axes)

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 1, "LogSumExp op requires exactly 1 input tensor"
        max_value = np.max(inputs[0], axis=self.axes, keepdims=True)
        shifted = inputs[0] - max_value
        sum_exp = np.sum(np.exp(shifted), axis=self.axes)
        return [np.log(sum_exp) + np.reshape(max_value, sum_exp.shape)]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        input_tensor = tensor.inputs[0]
        axes = _normalize_axes(axes=self.axes, shape=input_tensor.shape)
        if axes is None:
            reduced_shape = (1,) * len(input_tensor.shape)
        else:
            reduced_shape = tuple(
                1 if axis in axes else dim for axis, dim in enumerate(input_tensor.shape)
            )

        broadcast_grad = broadcast_to(
            reshape(incoming_grad, shape=reduced_shape), shape=input_tensor.shape
        )
        broadcast_output = broadcast_to(
            reshape(tensor, shape=reduced_shape), shape=input_tensor.shape
        )
        softmax = exp(input_tensor - broadcast_output)
        return [broadcast_grad * softmax]

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
        return Tensor(
            inputs=inputs,
            op=self,
            shape=self.infer_shape(inputs=inputs),
            label=label,
        )

    @override
    def infer_shape(self, inputs: list[Tensor]) -> tuple[int, ...]:
        assert len(inputs) == 1, "ZerosLike op requires exactly 1 input tensor"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 1, "ZerosLike op requires exactly 1 input tensor"
        return [np.zeros_like(inputs[0])]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [zeros_like(tensor.inputs[0])]

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
        return Tensor(
            inputs=inputs,
            op=self,
            shape=self.infer_shape(inputs=inputs),
            label=label,
        )

    @override
    def infer_shape(self, inputs: list[Tensor]) -> tuple[int, ...]:
        assert len(inputs) == 1, "OnesLike op requires exactly 1 input tensor"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 1, "OnesLike op requires exactly 1 input tensor"
        return [np.ones_like(inputs[0])]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [zeros_like(tensor.inputs[0])]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Reshape(TensorOp):
    shape: tuple[int, ...]

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

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
        assert len(inputs) == 1, "Reshape op requires exactly 1 input tensor"
        assert np.prod(inputs[0].shape) == np.prod(self.shape), "Reshape cannot change tensor size"
        return self.shape

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 1, "Reshape op requires exactly 1 input tensor"
        assert np.prod(inputs[0].shape) == np.prod(self.shape), "Reshape cannot change tensor size"
        return [np.reshape(inputs[0], self.shape)]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [reshape(incoming_grad, shape=tensor.inputs[0].shape)]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class BroadcastTo(TensorOp):
    shape: tuple[int, ...]

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

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
        assert len(inputs) == 1, "BroadcastTo op requires exactly 1 input tensor"
        _assert_broadcastable(input_shape=inputs[0].shape, target_shape=self.shape)
        return self.shape

    @override
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        assert len(inputs) == 1, "BroadcastTo op requires exactly 1 input tensor"
        _assert_broadcastable(input_shape=inputs[0].shape, target_shape=self.shape)
        return [np.broadcast_to(inputs[0], self.shape)]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        input_shape = tensor.inputs[0].shape
        padded_shape = (1,) * (len(tensor.shape) - len(input_shape)) + input_shape
        axes = tuple(axis for axis, dim in enumerate(padded_shape) if dim == 1)
        grad = summation(incoming_grad, axes=axes) if axes else incoming_grad
        return [reshape(grad, shape=input_shape)]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


def constant(value: float | int | ndarray, label: str | None = None) -> Tensor:
    """Create a constant Tensor with the given value and an optional label.
    This is used to denote constant values in the computational graph that are not supplied at
    evaluation time, but rather determined at graph construction time.
    """
    return Constant(value)(inputs=[], label=label)


def zeros(shape: tuple[int, ...], label: str | None = None) -> Tensor:
    """Create a constant tensor filled with zeros."""
    return constant(value=np.zeros(shape), label=label)


def ones(shape: tuple[int, ...], label: str | None = None) -> Tensor:
    """Create a constant tensor filled with ones."""
    return constant(value=np.ones(shape), label=label)


def input(shape: tuple[int, ...], label: str | None = None) -> Tensor:
    """Create an input Tensor with an optional label.
    This is used to denote the inputs to the computational graph.
    All input nodes must be assigned a
    value at evaluation time.
    """
    return Input(shape=shape)(inputs=[], label=label)


def _assert_broadcastable(input_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> None:
    assert len(input_shape) <= len(target_shape), "Cannot broadcast to a lower-rank shape"
    padded_shape = (1,) * (len(target_shape) - len(input_shape)) + input_shape
    for input_dim, target_dim in zip(padded_shape, target_shape):
        assert input_dim == 1 or input_dim == target_dim, "Input shape is not broadcastable"


def _broadcast_shape(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> tuple[int, ...]:
    rank = max(len(shape1), len(shape2))
    padded_shape1 = (1,) * (rank - len(shape1)) + shape1
    padded_shape2 = (1,) * (rank - len(shape2)) + shape2

    output_shape: list[int] = []
    for dim1, dim2 in zip(padded_shape1, padded_shape2):
        assert dim1 == dim2 or dim1 == 1 or dim2 == 1, "Input shapes are not broadcastable"
        output_shape.append(max(dim1, dim2))
    return tuple(output_shape)


def _broadcast_pair(t1: Tensor, t2: Tensor) -> tuple[Tensor, Tensor]:
    shape = _broadcast_shape(shape1=t1.shape, shape2=t2.shape)
    if t1.shape != shape:
        t1 = broadcast_to(t1, shape=shape)
    if t2.shape != shape:
        t2 = broadcast_to(t2, shape=shape)
    return t1, t2


def add(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    """Add two tensors elementwise, returning a new tensor that represents the output of this
    operation."""
    t1, t2 = _broadcast_pair(t1=t1, t2=t2)
    return Add()(inputs=[t1, t2], label=label)


def mul(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    """Multiply two tensors elementwise, returning a new tensor that represents the output of this
    operation."""
    t1, t2 = _broadcast_pair(t1=t1, t2=t2)
    return Mul()(inputs=[t1, t2], label=label)


def div(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    """Divide two tensors elementwise, returning a new tensor that represents the output of this
    operation."""
    t1, t2 = _broadcast_pair(t1=t1, t2=t2)
    return Div()(inputs=[t1, t2], label=label)


def mm(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    """Matrix multiply two tensors, returning a new tensor that represents the output of this
    operation."""
    return Matmul()(inputs=[t1, t2], label=label)


def power(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    """Raise one tensor elementwise to another tensor's power."""
    t1, t2 = _broadcast_pair(t1=t1, t2=t2)
    return Pow()(inputs=[t1, t2], label=label)


def transpose(
    t: Tensor,
    axes: tuple[int, int] | None = None,
    label: str | None = None,
) -> Tensor:
    """Transpose a tensor, returning a new tensor that represents the output of this operation."""
    return Transpose(axes=axes)(inputs=[t], label=label)


def reshape(t: Tensor, shape: tuple[int, ...], label: str | None = None) -> Tensor:
    """Reshape a tensor, returning a new tensor that represents the output of this operation."""
    return Reshape(shape=shape)(inputs=[t], label=label)


def broadcast_to(t: Tensor, shape: tuple[int, ...], label: str | None = None) -> Tensor:
    """Broadcast a tensor to a target shape as an explicit graph operation."""
    return BroadcastTo(shape=shape)(inputs=[t], label=label)


def summation(
    t: Tensor,
    axes: tuple[int, ...] | int | None = None,
    label: str | None = None,
) -> Tensor:
    """Sum a tensor across axes, returning a new tensor for the reduced output."""
    return Summation(axes=axes)(inputs=[t], label=label)


def negate(t: Tensor, label: str | None = None) -> Tensor:
    """Negate a tensor elementwise."""
    return Negate()(inputs=[t], label=label)


def exp(t: Tensor, label: str | None = None) -> Tensor:
    """Exponentiate a tensor elementwise."""
    return Exp()(inputs=[t], label=label)


def log(t: Tensor, label: str | None = None) -> Tensor:
    """Take the natural log of a tensor elementwise."""
    return Log()(inputs=[t], label=label)


def tanh(t: Tensor, label: str | None = None) -> Tensor:
    """Apply tanh to a tensor elementwise."""
    return Tanh()(inputs=[t], label=label)


def logsumexp(
    t: Tensor,
    axes: tuple[int, ...] | int | None = None,
    label: str | None = None,
) -> Tensor:
    """Compute a numerically stable log-sum-exp reduction."""
    return LogSumExp(axes=axes)(inputs=[t], label=label)


def zeros_like(t: Tensor, label: str | None = None) -> Tensor:
    """Create a tensor of zeros with the same shape as the input tensor."""
    return ZerosLike()(inputs=[t], label=label)


def ones_like(t: Tensor, label: str | None = None) -> Tensor:
    """Create a tensor of ones with the same shape as the input tensor."""
    return OnesLike()(inputs=[t], label=label)
