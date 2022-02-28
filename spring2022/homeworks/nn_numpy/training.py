import numpy as np
from sklearn.metrics import accuracy_score
from torchvision.datasets import MNIST

from layers import LinearLayer, ReLU, SoftmaxCE, Graph


EPOCH = 0 # your value
LEARNING_RATE = 0 # your value


def to_numpy(x):
  x = np.array(x).flatten()
  return x / 255


train_dataset = MNIST(".", download=True, train=True, transform=to_numpy)
valid_dataset = MNIST(".", download=True, train=False, transform=to_numpy)


layers = [
    LinearLayer(), # your params here
    ReLU(),
    LinearLayer(), # your params here
    ReLU(),
    LinearLayer(), # your params here
    SoftmaxCE()
]

graph = Graph(layers=layers, learning_rate=LEARNING_RATE)


for epoch in range(EPOCH):

    for x, y in train_dataset:
        graph.forward(x)
        graph.backward(y)

    y_true = []
    y_pred = []

    for x, y in train_dataset:
        out = graph.forward(x)
        graph.backward(y)
        y_true.append(y)
        y_pred.append(out.argmax())

    score = accuracy_score(y_true, y_pred)
    print(f"epoch: {epoch + 1}/{EPOCH}, accuracy: {score:.3f}")
