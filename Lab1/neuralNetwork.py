import numpy as np

def format_shape(shape):
    return "x".join(map(str, shape)) if shape else "()"

class Node:
    def __repr__(self):
        return "<{} shape={} at {}>".format(
            type(self).__name__, format_shape(self.data.shape), hex(id(self)))

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

class layerNode(Node):
    def __init__(self, *parents):
        assert all(isinstance(parent, Node) for parent in parents), (
            'Function input should be {}, expected {}, got {}'.
            format(Node.__name__, Node.__name__, tuple(type(parent).__name__ for parent in parents))
        )
        self.parents = parents
        self.data = self._forward(*(parent.data for parent in parents))

class Layer(DataNode):
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

class Linear(layerNode):
    @staticmethod
    def _forward(*inputs:np.ndarray):
        assert len(inputs) == 2, (
            "Inputs should have two dimensions, expected 2, got {}".format(len(inputs))
        )
        assert inputs[0].ndim==2, (
            "First input should have two dimensions, expected 2, got {}".format(inputs[0].shape)
        )
        assert inputs[1].ndim==2, (
            "First input should have two dimensions, expected 2, got {}".format(inputs[1].shape)
        )
        assert inputs[0].shape[1]==inputs[1].shape[0], (
            "Second dimension of first input should match first dimension of second input, "
            "expected inputs[0].shape[1]==inputs[1].shape[0], got {} and {}".
            format(inputs[0].shape[1], inputs[1].shape[0])
        )
        return np.dot(inputs[0] ,inputs[1])

    @staticmethod
    def _backward(gradient, *inputs):
        assert gradient.shape[0] == inputs[0].shape[0]
        assert gradient.shape[1] == inputs[1].shape[1]
        return [np.dot(gradient, inputs[1].T), np.dot(inputs[0].T, gradient)]


class add(layerNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2, (
            "Inputs should have two dimensions, expected 2, got {}".format(len(inputs))
        )
        assert inputs[0].ndim == 2, (
            "First input should have two dimensions, expected 2, got {}".format(inputs[0].shape)
        )
        assert inputs[1].ndim == 2, (
            "First input should have two dimensions, expected 2, got {}".format(inputs[1].shape)
        )
        assert inputs[0].shape == inputs[1].shape, (
            "Dimension of first input should match second input, "
            "expected inputs[0].shape==inputs[1].shape, got {} and {}".
            format(inputs[0].shape, inputs[1].shape)
        )
        return inputs[0] + inputs[1]

    @staticmethod
    def _backward(gradient, *inputs):
        assert gradient.shape == inputs[0].shape
        return [gradient, gradient]
class addBias(layerNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2, (
            "Inputs should have two dimensions, expected 2, got {}".format(len(inputs))
        )
        assert inputs[0].ndim == 2, (
            "First input should have two dimensions, expected 2, got {}".format(inputs[0].shape)
        )
        assert inputs[1].ndim == 2, (
            "First input should have two dimensions, expected 2, got {}".format(inputs[1].shape)
        )
        assert inputs[0].shape[1] == inputs[1].shape[1], (
            "Dimension of first input should match second input, "
            "expected inputs[0].shape==inputs[1].shape, got {} and {}".
            format(inputs[0].shape, inputs[1].shape)
        )
        return inputs[0] + inputs[1]

    @staticmethod
    def _backward(gradient, *inputs):
        assert gradient.shape == inputs[0].shape
        return [
            gredient,
            np.sum(gredient, axis=0, keepdims=True)
        ]

class ReLu(layerNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 1, (
            "Inputs should have one dimensions, expected 1, got {}".format(len(inputs))
        )
        assert inputs[0].ndim == 2, (
            "First input should have two dimensions, expected 2, got {}".format(inputs[0].shape)
        )
        return np.maximum(0, inputs[0])

    @staticmethod
    def _backward(gredient:np.ndarray, *inputs:np.ndarray):
        assert gredient.shape==inputs[0].shape
        return [gredient * np.where(inputs[0] > 0, 1, 0)]

class meanSquareLoss(layerNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2, (
            "Inputs should have two dimensions, expected 2, got {}".format(len(inputs))
        )
        assert inputs[0].ndim == 1, (
            "First input should have one dimensions, expected 1, got {}".format(inputs[0].shape)
        )
        assert inputs[1].ndim == 1, (
            "First input should have one dimensions, expected 1, got {}".format(inputs[1].shape)
        )
        assert inputs[0].shape == inputs[1].shape, (
            "Dimension of first input should match second input, "
            "expected inputs[0].shape==inputs[1].shape, got {} and {}".
            format(inputs[0].shape, inputs[1].shape)
        )
        return np.mean(np.square(inputs[0] - inputs[1])) / 2
    @staticmethod
    def _backward(gradient, *inputs):
        assert np.asarray(gradient).ndim == 0
        return [
            gradient * (inputs[0] - inputs[1]) / inputs[0].size,
            gradient * (inputs[1] - inputs[0]) / inputs[0].size
        ]

class softmaxLoss(layerNode):
    @staticmethod
    def _forward(*inputs:np.ndarray):
        assert len(inputs) == 2, (
            "Inputs should have two dimensions, expected 2, got {}".format(len(inputs))
        )
        assert inputs[0].ndim == 2, (
            "First input should have two dimensions, expected 1, got {}".format(inputs[0].shape)
        )
        assert inputs[1].ndim == 2, (
            "First input should have two dimensions, expected 1, got {}".format(inputs[1].shape)
        )
        assert np.all(inputs[1] >= 0), (
            "All probs in the labels input must be non-negative"
        )
        assert np.allclose(np.sum(inputs[1], axis=1), 1), (
            "Labels input must sum to 1 along each row"
        )
        assert inputs[0].shape == inputs[1].shape, (
            "Dimension of first input should match second input, "
            "expected inputs[0].shape==inputs[1].shape, got {} and {}".
            format(inputs[0].shape, inputs[1].shape)
        )
        predict = inputs[0]
        log_prob = predict - np.max(predict,axis=1, keepdims=True)
        log_prob = np.log(np.sum(np.exp(log_prob), axis=1, keepdims=True)) - log_prob
        return np.sum(log_prob * inputs[1]) / predict.shape[0]
    @staticmethod
    def _backward(gradient, *inputs):
        assert np.asarray(gradient).ndim == 0
        predict = inputs[0]
        log_prob = predict - np.max(predict, axis=1, keepdims=True)
        log_prob = -np.log(np.sum(np.exp(log_prob), axis=1, keepdims=True)) + log_prob
        return [
            gradient * (np.exp(log_prob) - inputs[1]) / inputs[0].shape[0],
            gradient * (-log_prob) / inputs[0].shape[0]
        ]

def gredient(loss, layers):
    assert isinstance(loss ,(meanSquareLoss, softmaxLoss)), (
        "Loss should be type of meanSquareLoss or softmaxLoss, got {!r}".format(type(loss).__name__)
    )
    assert all(isinstance(layer, Layer) for layer in layers), (
        'all layers should be type of {}, got {!r}'.format(
            Layer.__name__,
            tuple(type(layer).__name__ for layer in layers))
    )
    assert not hasattr(loss, "used")

    loss.used = True
    nodes = set()
    tape = []
    def goThroughNode(node):
        if node not in nodes:
            for parent in node.parents:
                goThroughNode(parent)
            nodes.add(node)
            tape.append(node)

    goThroughNode(loss)
    nodes |= set(layers)
    grads = {node: np.zeros_like(node.data) for node in nodes}
    grads[loss] = 1.0

    for node in reversed(tape):
        parent_grads = node._backward(
            grads[node], *(parent.data for parent in node.parents))
        for parent, parent_grad in zip(node.parents, parent_grads):
            grads[parent] += parent_grad

    return [Constant(grads[layer]) for layer in layers]