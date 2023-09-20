import numpy as np

def format_shape(shape):
    return "x".join(map(str, shape)) if shape else "()"

class Node:
    pass

class DataNode(Node):
    def __init__(self, data):
        self.parents = []
        self.data = data

    def _forward(self, *inputs):
        return self.data

    @staticmethod
    def _backward(gradient, *inputs):
        return []

class Constant(DataNode):
    def __init__(self, data):
        assert isinstance(data, np.ndarray), (
            "Data should be a numpy array, instead has type {!r}".format(
                type(data).__name__))
        assert np.issubdtype(data.dtype, np.floating), (
            "Data should be a float array, instead has data type {!r}".format(
                data.dtype))
        super().__init__(data)

class Layer:
    def __init__(self, *shape):
        assert len(shape) == 2, (
            f"Shape must have two dimensions, expected 2, got {len(shape)}"
        )
        assert all(isinstance(dim, int) and dim > 0 for dim in shape), (
            "Shape must have two positive integers, expected int, got {!r}".format(shape)
        )
        edge = np.sqrt(10 / np.mean(shape))
        data = np.random.uniform(high=edge, low=edge, size=shape)
        super().__init__(data)

    def update(self, direction, multiplier):
        assert isinstance(direction, Constant), (
            "Update direction should be {}, expected Constant, got {!r}".
            format(Constant.__name__,  type(direction).__name__)
        )
        assert direction.data.shape == self.data.shape, (
            "Update direction shape {} does not match parameter shape "
            "{}".format(
                format_shape(direction.data.shape),
                format_shape(self.data.shape)))
        assert isinstance(multiplier, (int, float)), (
            "Update multiplier should be a Python scaler, expected int or float, got {!r}".
            format(type(multiplier).__name__)
        )
        self.data += multiplier * direction.data
        assert np.all(np.isfinite(self.data)), (
            "Parameter contains NaN or infinity after update, cannot continue")

class
