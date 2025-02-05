import numpy as np

class Optimizer:
    def __init__(self):
        # Debug: Initializing Optimizer base
        self.regularizer = None

    def add_regularizer(self, reg):
        # Debug: Adding regularizer to optimizer
        self.regularizer = reg


class Sgd(Optimizer):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def calculate_update(self, weights, grads):
        # Debug: Sgd calculate_update called
        if self.regularizer is None:
            reg_term = 0
        else:
            reg_term = self.lr * self.regularizer.calculate_gradient(weights)

        # print("SGD grads:", grads)
        return weights - self.lr * grads - reg_term


class SgdWithMomentum(Optimizer):

    def __init__(self, lr, momentum_val):
        super().__init__()
        self.lr = lr
        self.momentum_val = momentum_val
        self.prev_update = 0

    def calculate_update(self, weights, grads):
        # Debug: SgdWithMomentum calculate_update
        if self.regularizer is None:
            reg_term = 0
        else:
            reg_term = self.lr * self.regularizer.calculate_gradient(weights)

        # Compute velocity term
        self.prev_update = self.momentum_val * self.prev_update - self.lr * grads
        return weights + self.prev_update - reg_term


class Adam(Optimizer):
    def __init__(self, lr, beta1, beta2):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        # Debug: Initializing Adam
        self.current_grad = 0
        self.first_moment = 0
        self.second_moment = 0
        self.iter_count = 1

    def calculate_update(self, weights, grads):
        # Debug: Adam calculate_update start
        self.current_grad = grads
        self.first_moment = self.beta1 * self.first_moment + (1 - self.beta1) * self.current_grad
        self.second_moment = self.beta2 * self.second_moment + (1 - self.beta2) * np.multiply(self.current_grad, self.current_grad)

        # Bias-corrected estimates
        first_moment_hat = self.first_moment / (1 - self.beta1 ** self.iter_count)
        second_moment_hat = self.second_moment / (1 - self.beta2 ** self.iter_count)
        self.iter_count += 1

        # Regularization component, if any
        if self.regularizer is None:
            reg_term = 0
        else:
            reg_term = self.lr * self.regularizer.calculate_gradient(weights)

        # Final Adam update
        return weights - self.lr * (
            first_moment_hat / (np.sqrt(second_moment_hat) + np.finfo(float).eps)
        ) - reg_term
