import numpy as np


class Node:
    def __init__(self, inputs=None, name=None):
        if inputs is None:
            inputs = []
        self.inputs = inputs
        self.outputs = []
        self.name = name
        self.value = None
        self.gradients = {}
        for node in inputs:
            node.outputs.append(self)

    def forward(self):
        raise NotImplemented

    def backward(self):
        raise NotImplemented

    def __repr__(self):
        return "Node(name={})".format(self.name)


class Placeholder(Node):
    def __init__(self, name=None):
        Node.__init__(self, name=name)

    def forward(self, value=None):
        if value is not None:
            self.value = value
            # It's a input node, when need to forward, this node initiate self's value.

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self] = grad_cost * 1

    def __repr__(self):
        return "Placeholder(name={})".format(self.name)


class Add(Node):
    def __init__(self, *nodes):
        Node.__init__(self, nodes)

    def forward(self):
        self.value = sum(map(lambda n: n.value, self.inputs))


class Linear(Node):
    def __init__(self, nodes, weights, bias, name=None):
        Node.__init__(self, inputs=[nodes, weights, bias], name=name)

    def forward(self):
        inputs = self.inputs[0].value
        weights = self.inputs[1].value
        bias = self.inputs[2].value
        self.value = np.dot(inputs, weights) + bias

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}

        for n in self.outputs:
            grad_cost = n.gradients[self]

            self.gradients[self.inputs[0]] = np.dot(grad_cost, self.inputs[1].value.T)
            self.gradients[self.inputs[1]] = np.dot(self.inputs[0].value.T, grad_cost)
            self.gradients[self.inputs[2]] = np.sum(grad_cost, axis=0, keepdims=False)

    def __repr__(self):
        return "Linear(name={})".format(self.name)


class Sigmoid(Node):
    def __init__(self, x, name=None):
        Node.__init__(self, inputs=[x], name=name)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        self.x = self.inputs[0].value
        self.value = self._sigmoid(self.x)

    def backward(self):
        self.partial = self._sigmoid(self.x) * (1 - self._sigmoid(self.x))

        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.inputs[0]] = grad_cost * self.partial
            # use * to keep all the dimension same!

    def __repr__(self):
        return "Sigmoid(name={})".format(self.name)


class MSE(Node):
    def __init__(self, y, y_hat, name=None):
        Node.__init__(self, inputs=[y, y_hat], name=name)

    def forward(self):
        y = self.inputs[0].value.reshape(-1, 1)
        y_hat = self.inputs[1].value.reshape(-1, 1)
        assert (y.shape == y_hat.shape)
        self.m = self.inputs[0].value.shape[0]
        self.diff = y - y_hat

        self.value = np.mean(self.diff ** 2)

    #         print("loss: {}".format(self.value))

    def backward(self):
        self.gradients[self.inputs[0]] = (2 / self.m) * self.diff
        self.gradients[self.inputs[1]] = (-2 / self.m) * self.diff

    def __repr__(self):
        return "MSE(name={})".format(self.name)
