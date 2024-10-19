from typing import List, Tuple

import numpy as np
from tqdm import tqdm


class GradSearch:
    def __init__(self, a: float = 1.0, b: float = 0.0, x0: float = 1.0, lr: float = 1e-2) -> None:
        """Initialize the GradSearch class with parameters for gradient search.

        :param a: Initial value for parameter a
        :param b: Initial value for parameter b
        :param x0: Initial value for parameter x0
        :param lr: Learning rate
        """
        self.a = a
        self.b = b
        self.x0 = x0
        self.l = lr
        self.loss: List[float] = []

    @staticmethod
    def f(x: np.ndarray, y: np.ndarray, a: float, b: float, x0: float) -> float:
        """Calculate the loss function.

        :param x: Input data x
        :param y: Input data y
        :param a: Parameter a
        :param b: Parameter b
        :param x0: Parameter x0
        :return: Calculated loss
        """
        return 0.5 * np.power(y - a * np.exp(-x / x0) - b, 2).sum()

    @staticmethod
    def dfda(x: np.ndarray, y: np.ndarray, a: float, b: float, x0: float) -> float:
        """Calculate the partial derivative of the loss function with respect to a.

        :param x: Input data x
        :param y: Input data y
        :param a: Parameter a
        :param b: Parameter b
        :param x0: Parameter x0
        :return: Partial derivative with respect to a
        """
        return ((a * np.exp(-x / x0) + b - y) * np.exp(-x / x0)).sum()

    @staticmethod
    def dfdb(x: np.ndarray, y: np.ndarray, a: float, b: float, x0: float) -> float:
        """Calculate the partial derivative of the loss function with respect to b.

        :param x: Input data x
        :param y: Input data y
        :param a: Parameter a
        :param b: Parameter b
        :param x0: Parameter x0
        :return: Partial derivative with respect to b
        """
        return (a * np.exp(-x / x0) + b - y).sum()

    @staticmethod
    def dfdx0(x: np.ndarray, y: np.ndarray, a: float, b: float, x0: float) -> float:
        """Calculate the partial derivative of the loss function with respect to x0.

        :param x: Input data x
        :param y: Input data y
        :param a: Parameter a
        :param b: Parameter b
        :param x0: Parameter x0
        :return: Partial derivative with respect to x0
        """
        return ((a * np.exp(-x / x0) + b - y) * a * np.exp(-x / x0) * x / pow(x0, 2)).sum()

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1_000) -> None:
        """Fit the model using gradient descent.

        :param x: Input data x
        :param y: Input data y
        :param epochs: Number of epochs for training
        """
        self.loss.append(self.f(x, y, self.a, self.b, self.x0))
        for _ in tqdm(range(epochs)):
            self.a -= self.l * self.dfda(x, y, self.a, self.b, self.x0)
            self.b -= self.l * self.dfdb(x, y, self.a, self.b, self.x0)
            self.x0 -= self.l * self.dfdx0(x, y, self.a, self.b, self.x0)
            self.loss.append(self.f(x, y, self.a, self.b, self.x0))

    def get_a_b_x0(self) -> Tuple[float, float, float]:
        """Get the current values of a, b, and x0.

        :return: Tuple containing a, b, and x0
        """
        return self.a, self.b, self.x0
