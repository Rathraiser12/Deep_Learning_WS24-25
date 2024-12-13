import numpy as np
from sympy.abc import epsilon


class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = np.atleast_1d(weight_tensor)
        gradient_tensor = np.atleast_1d(gradient_tensor)
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights


class SgdWithMomentum:
    def __init__(self, learning_rate: float, momentum_rate: float):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = np.atleast_1d(weight_tensor)
        gradient_tensor = np.atleast_1d(gradient_tensor)
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        updated_weights = weight_tensor + self.velocity
        return updated_weights


class Adam:
    def __init__(self, learning_rate: float, mu: float, rho: float):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-8
        self.m_t = None
        self.v_t = None
        self.t = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        weight_tensor = np.atleast_1d(weight_tensor)
        gradient_tensor = np.atleast_1d(gradient_tensor)
        if self.m_t is None or self.v_t is None:
            self.m_t = np.zeros_like(weight_tensor)
            self.v_t = np.zeros_like(weight_tensor)

        self.t += 1

        self.m_t = self.mu * self.m_t + (1 - self.mu) * gradient_tensor
        self.v_t = self.rho * self.v_t + (1 - self.rho) * (gradient_tensor ** 2)

        m_t_hat = self.m_t / (1 - self.mu ** self.t)
        v_t_hat = self.v_t / (1 - self.rho ** self.t)

        updated_weights = weight_tensor - self.learning_rate * m_t_hat / (np.sqrt(v_t_hat) + self.epsilon)
        return updated_weights
