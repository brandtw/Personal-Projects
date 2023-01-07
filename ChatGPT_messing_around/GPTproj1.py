import numpy as np

class Bandit:
  def __init__(self, p):
    # p is the probability of the bandit paying out a reward
    self.p = p
    
  def pull(self):
    # Returns a reward of 1 with probability p and 0 with probability 1-p
    return np.random.random() < self.p

class Agent:
  def __init__(self, n_bandits):
    # Initialize the agent with a count of the number of bandits
    self.bandits = [Bandit(np.random.random()) for _ in range(n_bandits)]
    self.n_bandits = n_bandits
  
  def select_bandit(self):
    # Select a random bandit
    return np.random.randint(self.n_bandits)
  
  def learn(self, n_steps):
    # Play the multi-armed bandit problem for n_steps steps
    rewards = np.zeros(self.n_bandits)
    counts = np.zeros(self.n_bandits)
    for i in range(n_steps):
      bandit = self.select_bandit()
      reward = self.bandits[bandit].pull()
      rewards[bandit] += reward
      counts[bandit] += 1
      
    # Calculate the average reward for each bandit
    return rewards / counts

# Create an agent with 10 bandits
agent = Agent(10)

# Have the agent learn for 1000 steps
rewards = agent.learn(1000)

# Print the average reward for each bandit
print(rewards)
