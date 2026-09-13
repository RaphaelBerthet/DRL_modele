import numpy as np

def relu(z: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    return np.maximum(0, z)


def relu_derivative(z: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    return (z > 0).astype(np.float64)