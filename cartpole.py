import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

from agent import DQNAgent
from loss import huber_loss
from memory import Memory
from policy import EpsilonGreedyDecayPolicy
from gym_helper import GymHelper


def build_model(num_states, num_actions, alpha=0.001):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=num_states))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss=huber_loss, optimizer=RMSprop(lr=alpha))
    return model


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    ALPHA = 0.00025
    MEMORY_LEN = 100000
    EPSILON_MAX = 1.0
    EPSILON_MIN = 0.01
    LAMBDA = 0.00095
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPISODES = 1000
    UPDATE_TARGET_FREQUENCY = 1000

    model = build_model(num_states, num_actions, ALPHA)
    target_model = build_model(num_states, num_actions, ALPHA)

    memory = Memory(MEMORY_LEN)
    policy = EpsilonGreedyDecayPolicy(EPSILON_MAX, EPSILON_MIN, LAMBDA)

    agent = DQNAgent(num_states, num_actions, model, target_model, memory, policy, BATCH_SIZE, GAMMA)

    helper = GymHelper(env, agent, EPISODES, UPDATE_TARGET_FREQUENCY)

    try:
        helper.run(verbose=True)
    finally:
        agent.model.save('model_{}_{}.h5'.format(helper.env_name, agent.__class__.__name__))

    with open('episode_rewards_{}_{}.txt'.format(helper.env_name, agent.__class__.__name__), 'w') as f:
        f.write("\n".join(map(str, helper.episode_rewards)))
