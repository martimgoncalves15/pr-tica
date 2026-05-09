import gymnasium as gym
import numpy as np
import random
import tabular_sarsa as Tabular_SARSA
import matplotlib.pyplot as plt
import sys


class SARSA(Tabular_SARSA.Tabular_SARSA):
    def __init__(self):
        super(SARSA, self).__init__()

    # TODO
    def learn_policy(
        self, env, gamma, learning_rate, epsilon, lambda_value, num_episodes
    ):
        """
        Implement Sarsa-lambda algorithm to update qtable, etable and learning policy.
        Input:
            all parameters

        Output:
            This function returns the updated qtable, learning policy and the reward after each episode.

        """
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.lambda_value = lambda_value
        rewards_each_learning_episode = []
        for i in range(num_episodes):
            state = env.reset()[0]
            action = self.LearningPolicy(state)
            episodic_reward = 0
            self.etable = np.zeros((self.num_states, self.num_actions))
            while True:
                next_state, reward, done, _, info = env.step(action)  # take a random action

            "*** Preencha o resto do algoritmo!!***"

        np.save("qvalues_taxi_sarsa_lambda", self.qtable)
        np.save("policy_taxi_sarsa_lambda", self.policy)

        return self.policy, self.qtable, rewards_each_learning_episode

    def LearningPolicy(self, state, testing=False):
        return Tabular_SARSA.Tabular_SARSA.learningPolicy(self, state, testing=testing)

def plot_rewards(episode_rewards):
    """
    Plots a learning curve for SARSA

    Input:
        episode_rewards: a list of episode rewards

    """
    plt.plot(episode_rewards)
    plt.ylabel("rewards per episode")
    plt.ion()
    plt.savefig("rewards_plot_lambda.png")

def render_visualization(learned_policy):
    """
    Renders a taxi problem visualization

    Input:
        learned_policy: the learned SARSA policy to be used by the taxi

    """
    env = gym.make("Taxi-v3", render_mode="human")
    state = env.reset()[0]
    env.render()
    while True:
        next_state, reward, done, _, info = env.step(learned_policy[state, 0])
        env.render()
        print("Reward: {}".format(reward))
        state = next_state
        if done:
            break
    env.close()

def avg_episode_rewards(num_runs):
    """
    Runs the learner algorithms a number of times and averages the episodic rewards
    from all runs for each episode

    Input:
        num_runs: the number of times to run the SARSA learner

    Output:
        episode_rewards: a list of averaged rewards per episode over a num_runs number of times
        learned_policy: the policy learned by the last run of sarsaLearner.learn_policy() to be
        used in problem visualization
    """
    episode_rewards = []
    for _ in range(num_runs):
        env = gym.make("Taxi-v3")
        env.reset()
        sarsaLearner = SARSA()
        learned_policy, q_values, single_run_er = sarsaLearner.learn_policy(
            env, 0.95, 0.2, 0.1, 0.8, 1000
        )         # single_run_er is the episodic reward for each run

        if not episode_rewards: # on the first iteration of this loop, episodeRewards will be empty
            episode_rewards = single_run_er
        else: # add this run's ERs to previous runs in order to calculate the average later
            episode_rewards = [episode_rewards[i] + single_run_er[i] for i in range(len(single_run_er))]

    # Get the average over ten runs
    episode_rewards = [er / num_runs for er in episode_rewards]

    return episode_rewards, learned_policy

def test_policy():
    policy = np.load("policy_taxi_sarsa_lambda_avaliacao.npy")
    env = gym.make("Taxi-v3", render_mode="human")
    env.reset()
    num_episodes = 10
    rewards_each_test_episode = []
    steps_each_test_episode = []

    for _ in range(num_episodes):
        state = env.reset()[0]
        episodic_reward = 0
        steps = 0
        while True:
            action = policy[state][0]
            next_state, reward, done, _, info = env.step(action)
            steps += 1

            episodic_reward += reward

            state = next_state

            if done or steps >= 200:
                break

        rewards_each_test_episode.append(episodic_reward)
        steps_each_test_episode.append(steps)

    print("Rewards each test episode: {}".format(rewards_each_test_episode))
    print("Steps each test episode: {}".format(steps_each_test_episode))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        "*** run without input arguments to learn ***"
        episode_rewards, learned_policy = avg_episode_rewards(10)
        plot_rewards(episode_rewards)
        render_visualization(learned_policy)
    elif sys.argv[1] == "test":
        "*** run to test the saved policy with input argument test ***"
        test_policy()
    else:
        print("unknown input argument")
