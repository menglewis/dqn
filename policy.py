import random
import math
import numpy as np


class Policy(object):
    def choose(self, q_values):
        raise NotImplementedError

    def decay(self):
        raise NotImplementedError


class GreedyPolicy(Policy):
    def choose(self, q_values):
        return np.argmax(q_values)

    def decay(self):
        pass


class EpsilonGreedyDecayPolicy(Policy):
    def __init__(self, epsilon_max, epsilon_min, lambda_):
        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.lambda_ = lambda_
        self.steps = 0

    def choose(self, q_values):
        if random.random() < self.epsilon:
            return random.randint(0, len(q_values)-1)
        else:
            return np.argmax(q_values)

    def decay(self):
        self.steps += 1
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.lambda_ * self.steps)
