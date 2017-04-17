from collections import deque


class GymHelper(object):
    """
    Helper class to add some convenience methods to help with training the agent as well
    as keeping track of performance throughout episodes.
    """

    def __init__(self, env, agent, episodes, update_target_frequency=1000):
        self.env = env
        self.env_name = env.spec.id
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.max_steps = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

        self.agent = agent
        self.episodes = episodes
        self.update_target_frequency = update_target_frequency

        self.episode_rewards = []
        self.last_100_rewards = deque(maxlen=100)
        self.total_steps = 0

    def run_episode(self, render=False):
        state = self.env.reset()
        episode_reward = 0
        for step in range(self.max_steps):
            if render:
                self.env.render()
            action = self.agent.act(state)
            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            self.agent.observe(state, action, reward, next_state, done)
            self.agent.replay()
            self.agent.policy.decay()

            state = next_state
            self.total_steps += 1
            if self.total_steps % self.update_target_frequency == 0:
                self.agent.update_target_model()
            if done:
                break
        return episode_reward

    def run(self, verbose=False, render=False):
        for episode in range(1, self.episodes + 1):
            episode_reward = self.run_episode(render=render)
            self.episode_rewards.append(episode_reward)
            self.last_100_rewards.append(episode_reward)

            if verbose:
                if len(self.last_100_rewards) == 100:
                    rolling_avg = sum(self.last_100_rewards)/len(self.last_100_rewards)
                    print("Episode {} | Reward {} | Last 100 {}".format(episode, episode_reward, rolling_avg))
                else:
                    print("Episode {} | Reward {}".format(episode, episode_reward))
        return self.episode_rewards
