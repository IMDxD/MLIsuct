import numpy as np
import seaborn as sns
from sklearn.datasets import make_circles


class CircleDataset:

    def __init__(self, n_samples):

        self.X, self.y = make_circles(n_samples=n_samples)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return {
            "points": self.X[index],
            "label": self.y[index]
        }

    def plot(self):
        sns.scatterplot(x=self.X[:, 0], y=self.X[:, 1], hue=self.y)

    def move_center(self, center_move_coor: np.ndarray, label: int):
        idx = np.where(self.y == label)[0]
        self.X[idx, :] += center_move_coor
