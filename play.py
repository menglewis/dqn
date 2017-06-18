import click
import gym
from keras.models import load_model

from agent import DQNAgent
from loss import huber_loss
from memory import Memory
from policy import GreedyPolicy


@click.command()
@click.option('--env-name', default='CartPole-v0', prompt='Env Name (CartPole-v0, LunarLander-v2, etc)')
@click.option('--model-path', prompt='Model path')
@click.option('--episodes', default=1, prompt='Number of Episodes')
@click.option('--max-steps', default=None, type=int, help='Max steps (defaults to environment max)')
def play(env_name, model_path, episodes, max_steps):
    env = gym.make(env_name)
    model = load_model(model_path, custom_objects={'huber_loss': huber_loss})
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    if max_steps is None:
        max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    memory = Memory(0)
    policy = GreedyPolicy()
    agent = DQNAgent(num_states, num_actions, model, model, memory, policy)

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            env.render()
            action = agent.act(state)

            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            state = next_state
            if done:
                break
        print("Reward: {}".format(episode_reward))

if __name__ == "__main__":
    play()
