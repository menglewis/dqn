import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
import click

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


@click.command()
@click.option('--alpha', default=0.00025)
@click.option('--memory-len', default=100000)
@click.option('--epsilon-max', default=1.0)
@click.option('--epsilon-min', default=0.01)
@click.option('--lambda_', default=0.00095)
@click.option('--batch-size', default=64)
@click.option('--gamma', default=0.99)
@click.option('--episodes', default=1000)
@click.option('--update-target-frequency', default=1000)
@click.option('--verbose', default=False, type=bool)
def run_cartpole(alpha, memory_len, epsilon_max, epsilon_min, lambda_, batch_size,
                 gamma, episodes, update_target_frequency, verbose):
    env = gym.make('CartPole-v0')
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = build_model(num_states, num_actions, alpha)
    target_model = build_model(num_states, num_actions, alpha)

    memory = Memory(memory_len)
    policy = EpsilonGreedyDecayPolicy(epsilon_max, epsilon_min, lambda_)

    agent = DQNAgent(num_states, num_actions, model, target_model, memory, policy, batch_size, gamma)

    helper = GymHelper(env, agent, episodes, update_target_frequency)

    try:
        helper.run(verbose=verbose)
    finally:
        hyperparameters = {
            'alpha': alpha,
            'memory_len': memory_len,
            'epsilon_max': epsilon_max,
            'epsilon_min': epsilon_min,
            'lambda_': lambda_,
            'batch_size': batch_size,
            'gamma': gamma,
            'episodes': episodes,
            'update_target_frequency': update_target_frequency,
        }
        helper.save(hyperparameters)


if __name__ == "__main__":
    run_cartpole()
