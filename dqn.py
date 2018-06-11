import random
import math
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque

np.random.seed(1)


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

        self.model = self._build_model()
        self.target_model = self._build_model()


    def _build_model(self):
        model = Sequential()
        state_shape  = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(
            loss="mean_squared_error",
            optimizer=Adam(lr=self.alpha,decay=self.alpha_decay)
        )
        print(model.summary())
        return model
    
    def decay_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        return self.epsilon

    def preprocess_state(self, state):
        return state.reshape(1,2)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])
    
    def choose_action(self, state):
        if np.random.random() < self.decay_epsilon():
            return np.random.choice([0, 1])
        else:
            return np.argmax(self.model.predict(state)[0])

    def replay(self):
        miniBatch = random.sample(
            self.memory, 
            min(len(self.memory), self.batch_size)
        )
        x_batch = []
        y_batch = []
        for state, action, reward, new_state, done in miniBatch:
            target = self.target_model.predict(state)
            if(done):
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + self.gamma * Q_future
            x_batch.append(state[0])
            y_batch.append(target[0])
        self.model.fit(np.array(x_batch), np.array(y_batch), epochs=1, verbose=0, batch_size=len(x_batch))

    def train_target(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)
