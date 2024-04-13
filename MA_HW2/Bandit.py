"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from logs import *
import random
import csv
import matplotlib.pyplot as plt

logging.basicConfig
logger = logging.getLogger("MAB Application")

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        self.true_means = p  # True means of each arm
        self.estimated_means = [0.0] * len(p)  # Estimated means of each arm
        self.action_counts = [0] * len(p)  # Number of times each arm is pulled

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass


# --------------------------------------#


import random
import numpy as np

class EpsilonGreedy(Bandit):
    def __init__(self, p, epsilon=0.1):
        super().__init__(p)
        self.epsilon = epsilon
        self.true_means = p
        self.action_counts = [0] * len(p)
        self.action_values = [0.0] * len(p)

    def __repr__(self):
        return f"EpsilonGreedy Bandit with epsilon={self.epsilon}"

    def pull(self):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.true_means) - 1)
        else:
            return self.action_values.index(max(self.action_values))

    def update(self, arm, reward):
        self.action_counts[arm] += 1
        n = self.action_counts[arm]
        self.action_values[arm] += (1 / n) * (reward - self.action_values[arm])

    def experiment(self, num_trials):
        rewards = []
        for _ in range(num_trials):
            arm = self.pull()
            reward = self.true_means[arm]
            self.update(arm, reward)
            rewards.append(reward)
        return rewards

    def report(self):
        avg_reward = sum(self.action_values) / len(self.action_values)
        avg_regret = max(self.true_means) - avg_reward
        return f"EpsilonGreedy Results: Average Reward={avg_reward:.2f}, Average Regret={avg_regret:.2f}"


class ThompsonSampling(Bandit):
    def __init__(self, p):
        super().__init__(p)
        self.alpha = [1] * len(p)
        self.beta = [1] * len(p)

    def __repr__(self):
        return "ThompsonSampling Bandit"

    def pull(self):
        sampled_means = [random.betavariate(self.alpha[i], self.beta[i]) for i in range(len(self.true_means))]
        return sampled_means.index(max(sampled_means))

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def experiment(self, num_trials):
        rewards = []
        for _ in range(num_trials):
            arm = self.pull()
            reward = self.true_means[arm]
            self.update(arm, reward)
            rewards.append(reward)
        return rewards

    def report(self):
        avg_reward = sum(self.alpha) / (sum(self.alpha) + sum(self.beta))
        avg_regret = max(self.true_means) - avg_reward
        return f"ThompsonSampling Results: Average Reward={avg_reward:.2f}, Average Regret={avg_regret:.2f}"


Bandit_Reward = [1, 2, 3, 4]
NumberOfTrials: 20000


class Visualization:

    def plot1(self, epsilon_greedy_rewards, thompson_rewards):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        cumulative_epsilon_greedy_rewards = np.cumsum(epsilon_greedy_rewards)
        cumulative_thompson_rewards = np.cumsum(thompson_rewards)
        plt.plot(cumulative_epsilon_greedy_rewards, label="Epsilon-Greedy")
        plt.plot(cumulative_thompson_rewards, label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(np.log(cumulative_epsilon_greedy_rewards), label="Epsilon-Greedy (log scale)")
        plt.plot(np.log(cumulative_thompson_rewards), label="Thompson Sampling (log scale)")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward (log scale)")
        plt.legend()

        plt.show()

    def plot2(self, epsilon_greedy_rewards, thompson_rewards):

        plt.figure(figsize=(10, 5))
        cumulative_epsilon_greedy_rewards = [sum(epsilon_greedy_rewards[:i + 1]) for i in
                                             range(len(epsilon_greedy_rewards))]
        cumulative_thompson_rewards = [sum(thompson_rewards[:i + 1]) for i in range(len(thompson_rewards))]

        plt.plot(range(len(cumulative_epsilon_greedy_rewards)), cumulative_epsilon_greedy_rewards,
                 label="Epsilon-Greedy")
        plt.plot(range(len(cumulative_thompson_rewards)), cumulative_thompson_rewards, label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Rewards")
        plt.legend()
        plt.show()

    def store_rewards_to_csv(self, epsilon_greedy_rewards, thompson_rewards):

        with open('bandit_rewards.csv', mode='w', newline='') as csv_file:
            fieldnames = ['Bandit', 'Reward', 'Algorithm']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for reward in epsilon_greedy_rewards:
                writer.writerow({'Bandit': 'EpsilonGreedy', 'Reward': reward, 'Algorithm': 'Epsilon-Greedy'})

            for reward in thompson_rewards:
                writer.writerow({'Bandit': 'ThompsonSampling', 'Reward': reward, 'Algorithm': 'Thompson Sampling'})

    def report_cumulative_reward_and_regret(self, epsilon_greedy_rewards, thompson_rewards):

        cumulative_epsilon_greedy_reward = sum(epsilon_greedy_rewards)
        cumulative_thompson_reward = sum(thompson_rewards)
        cumulative_epsilon_greedy_regret = max(Bandit_Reward) * len(
            epsilon_greedy_rewards) - cumulative_epsilon_greedy_reward
        cumulative_thompson_regret = max(Bandit_Reward) * len(thompson_rewards) - cumulative_thompson_reward

        print(f'Cumulative Reward - Epsilon-Greedy: {cumulative_epsilon_greedy_reward}')
        print(f'Cumulative Reward - Thompson Sampling: {cumulative_thompson_reward}')
        print(f'Cumulative Regret - Epsilon-Greedy: {cumulative_epsilon_greedy_regret}')
        print(f'Cumulative Regret - Thompson Sampling: {cumulative_thompson_regret}')


def comparison(num_trials):
    epsilon = 0.1
    epsilon_greedy_bandit = EpsilonGreedy(Bandit_Reward, epsilon)
    thompson_bandit = ThompsonSampling(Bandit_Reward)

    epsilon_greedy_rewards = epsilon_greedy_bandit.experiment(num_trials)
    thompson_rewards = thompson_bandit.experiment(num_trials)

    vis = Visualization()

    vis.plot1(epsilon_greedy_rewards, thompson_rewards)

    vis.plot2(epsilon_greedy_rewards, thompson_rewards)

    vis.store_rewards_to_csv(epsilon_greedy_rewards, thompson_rewards)

    vis.report_cumulative_reward_and_regret(epsilon_greedy_rewards, thompson_rewards)


if __name__ == "__main__":
    num_trials = 20000
    comparison(num_trials)


def comparison():
    # think of a way to compare the performances of the two algorithms VISUALLY and
    pass


if __name__ == '__main__':
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
