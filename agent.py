import numpy as np


class DQNAgent(object):
    def __init__(self, num_states, num_actions,
                 model, target_model, memory, policy,
                 batch_size=32, gamma=0.99):
        self.num_states = num_states
        self.num_actions = num_actions

        self.model = model
        self.target_model = target_model
        self.memory = memory
        self.policy = policy

        self.batch_size = batch_size
        self.gamma = gamma

    def act(self, state):
        return self.policy.choose(self.model.predict(state.reshape(1, self.num_states)).flatten())

    def observe(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def replay(self):
        minibatch = self.memory.sample(self.batch_size)

        states = np.array([memory[0] for memory in minibatch])
        next_states = np.array([(memory[3] if memory[3] is not None else np.zeros(self.num_states)) for memory in minibatch])

        predictions = self.model.predict(states)
        next_predictions = self.target_model.predict(next_states)
        x = np.zeros((len(minibatch), self.num_states))
        y = np.zeros((len(minibatch), self.num_actions))

        for i, memory in enumerate(minibatch):
            state, action, reward, next_state, terminal = memory

            target = predictions[i]
            if terminal:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.amax(next_predictions[i])
            x[i] = state
            y[i] = target

        self.model.fit(x, y, batch_size=self.batch_size, nb_epoch=1, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
