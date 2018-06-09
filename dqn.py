import random
import math
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque


class DQN:
    def __init__(
        self,
        env,
        memory_size,
        batch_size,
        gamma,
        epsilon,
        epsilon_min,
        epsilon_decay,
        alpha,
        alpha_decay,
    ):
        self.env = env
        self.memory  = deque(maxlen=memory_size)
        self.batch_size = batch_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.tau = 0.125

        self.model = self._build_model()
        self.target_model = self._build_model()

        self.cost_history = []

    def _build_model(self):
        model = Sequential()
        state_shape  = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.alpha)) #decay=self.alpha_decay)
        return model
    
    def _get_epsilon(self, episode):
        self.epsilon = min(self.epsilon, 1.0 - math.log10((episode + 1) * self.epsilon_decay))
        self.epsilon = max(self.epsilon_min, self.epsilon)
        return self.epsilon

    def preprocess_state(self, state):
        return np.reshape(state, [1, 2])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])
    
    def choose_action(self, state, episode):
        epsilon = self._get_epsilon(episode)
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            # 0?
            return np.argmax(self.model.predict(state)[0])

    def replay(self):
        miniBatch = random.sample(
            self.memory, 
            min(len(self.memory), self.batch_size)
        )

        for state, action, reward, new_state, done in miniBatch:
            target = self.target_model.predict(state)
            if(done):
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + self.gamma * Q_future

            self.model.fit(state, target, epochs=1, verbose=0)
            # self.cost_history.append(history.history['loss'])

    def train_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
