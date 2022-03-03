from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score
from torchvision.datasets import MNIST

from layers import LinearLayer, ReLU, SoftmaxCE, Graph


EPOCH = # your value
LEARNING_RATE = # your value


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
    start = perf_counter()
    idxs = np.arange(len(train_dataset))
    np.random.shuffle(idxs)
    for i in idxs:
        x, y = train_dataset[i]
        graph.forward(x)
        graph.backward(y)
        
    y_true = []
    y_pred = []

    for x, y in valid_dataset:
        out = graph.forward(x)
        y_true.append(y)
        y_pred.append(out.argmax())

    score = accuracy_score(y_true, y_pred)
    print(f"epoch: {epoch + 1}/{EPOCH}, accuracy: {score:.3f}")
