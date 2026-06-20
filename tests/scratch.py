import numpy as np
import tmlc

x = tmlc.input(shape=(2, 2), label="x")
y = tmlc.input(shape=(2, 2), label="y")
z = tmlc.input(shape=(2, 2), label="z")

a = x * y
b = a * z
c = b * a
out = c + a

output = tmlc.run(
    inputs={
        x: np.array([[1, 2], [3, 4]]),
        y: np.array([[5, 6], [7, 8]]),
        z: np.array([[1, 1], [1, 1]]),
    },
    outputs=[a, b, c],
)

print(output)

grads = tmlc.gradients(output_node=out, target_nodes=[a, b, c])
a_grad = grads[0]
b_grad = grads[1]
c_grad = grads[2]

output = tmlc.run(
    inputs={
        x: np.array([[1, 2], [3, 4]]),
        y: np.array([[5, 6], [7, 8]]),
        z: np.array([[1, 1], [1, 1]]),
    },
    outputs=[a_grad, b_grad, c_grad],
)

print(output)
