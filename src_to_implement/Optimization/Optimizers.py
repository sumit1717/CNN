import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor ):
        # returns the updated weights according to the basic gradient descent update scheme.
        updated_weights_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity_tensor = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        momentum_tensor = self.momentum_rate * self.velocity_tensor - self.learning_rate * gradient_tensor
        # print("----------momentum_tensormomentum_tensormomentum_tensor--------",weight_tensor, momentum_tensor, gradient_tensor, "----------momentum_tensormomentum_tensormomentum_tensor--------")
        self.velocity_tensor = momentum_tensor
        updated_weights_tensor_momentum = weight_tensor + momentum_tensor
        return updated_weights_tensor_momentum


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-8
        self.velocity_tensor = 0
        self.rate_tensor = 0
        self.exponent = 1
    def calculate_update(self, weight_tensor, gradient_tensor ):
        velocity_tensor = self.mu * self.velocity_tensor + (1-self.mu) * gradient_tensor
        rate_tensor = self.rho * self.rate_tensor + (1-self.rho) * (gradient_tensor ** 2)
        v_hat = velocity_tensor/(1-np.power(self.mu, self.exponent))
        r_hat = rate_tensor/(1-np.power(self.rho, self.exponent))

        self.exponent += 1
        self.velocity_tensor = velocity_tensor
        self.rate_tensor = rate_tensor

        updated_weights_tensor = weight_tensor - self.learning_rate * (v_hat/(np.sqrt(r_hat) + self.epsilon))

        return updated_weights_tensor
