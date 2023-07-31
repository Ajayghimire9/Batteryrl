#!/usr/bin/env python
# coding: utf-8

# In[26]:


#importing the libraries 

import gym
from gym import spaces
import numpy as np
import pandas as pd
from collections import namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pack_sequence




# Define the Transition namedtuple (Step 1)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class MicrogridEnv(gym.Env):
    def __init__(self, consumption_data, production_data,spotmarket_data, n_past=5):
        super().__init__()

        # Microgrid parameters
        self.s_B = 0  # Initial energy in the battery
        self.s_H2 = 0  # Initial energy in the hydrogen tank

        # Energy storage sizing for battery and hydrogen
        self.x_B = 500  # Wh
        self.x_H2 = 14000  # Wp

        # Battery discharge and charge efficiency
        self.eta_B = 0.9
        self.zeta_B = 0.9

        # Electrolysis and fuel cells efficiencies
        self.eta_H2 = 0.65
        self.zeta_H2 = 0.65

        # Reward function parameters
        self.k = 2  # cost endured per kWh not supplied within the microgrid
        self.k_H2 = 0.1  # revenue/cost per kWh of hydrogen produced/used

        self.n_past = n_past  # Number of past consumption and production values to include in the state

        # Consumption and production data
        self.consumption_data = consumption_data
        self.production_data = production_data

        # State vector
        self.state = torch.tensor([self.s_B, self.s_H2])

        # Define action and observation space
        low_bounds = [0] * (2 + 2 * n_past)
        high_bounds = [np.inf] * (2 + 2 * n_past)
        self.observation_space = spaces.Box(low=np.array(low_bounds), high=np.array(high_bounds))

        self.action_space = spaces.Discrete(2)  # Actions: (0) discharge, (1) charge

        # State vector (converted to PyTorch tensor)
        self.state = torch.tensor([self.s_B, self.s_H2], dtype=torch.float32)

        # Reset the environment upon initialization
        self.reset()

        
    def step(self, action):
        # Map the action to a specific storage operation
        if action == 0:  # discharge
            a_H2_t = -self.eta_H2
            self.s_B = max(self.s_B - self.zeta_B, 0)
            self.s_H2 = max(self.s_H2 + a_H2_t, 0)
        elif action == 1:  # charge
            a_H2_t = self.zeta_H2
            self.s_B = min(self.s_B + self.eta_B, self.x_B)
            self.s_H2 = min(self.s_H2 + a_H2_t, self.x_H2)
        else:
            raise ValueError("Invalid action. Actions must be 0 (discharge) or 1 (charge).")

        # Update past consumption and production
        self.past_consumption.pop(0)
        self.past_consumption.append(self.consumption_data[self.t])
        self.past_production.pop(0)
        self.past_production.append(self.production_data[self.t])

        # Update the state
        self.state = np.concatenate(([self.s_B, self.s_H2], self.past_consumption, self.past_production))


        # Calculate the net electricity demand d_t
        d_t = self.consumption_data[self.t] - self.production_data[self.t]

        # Calculate the power balance within the microgrid
        delta_t = -a_H2_t - d_t

        # Calculate reward
        r_H2 = self.k_H2 * a_H2_t if a_H2_t > 0 else 0
        r_minus = self.k * delta_t if delta_t < 0 else 0
        reward = r_H2 + r_minus

        # Increase time step
        self.t += 1

        # Signal the end of the episode after a fixed number of steps (e.g., 24 hours)
        done = self.t >= 24

        return self.state, reward, done, {}

    def reset(self):
        # Reset time step
        self.t = 0

        # Reset the short-term storage (battery) and long-term storage (hydrogen) to their initial states.
        self.s_B = self.x_B / 2  # Assuming battery starts half-charged
        self.s_H2 = self.x_H2 / 2  # Assuming hydrogen storage starts half-filled

        # Initialize past consumption and production as zero
        self.past_consumption = [0]*self.n_past
        self.past_production = [0]*self.n_past

        # Update the state
        self.state = np.array([self.s_B, self.s_H2] + self.past_consumption + self.past_production)

        # Return the initial state
        return self.state
    
    def render(self, mode='human'):
        print(f"State: {self.state}")

    def close(self):
        pass
    
# loading the datasets
pv_prod_test = np.load("C:\\Users\\ajayg\\Desktop\\University\\Thesis\\Datasets\\BelgiumPV_prod_test.npy")
pv_prod_train = np.load("C:\\Users\\ajayg\\Desktop\\University\\Thesis\\Datasets\\BelgiumPV_prod_train.npy")
nondet_cons_test = np.load("C:\\Users\\ajayg\\Desktop\\University\\Thesis\\Datasets\\example_nondeterminist_cons_test.npy")
nondet_cons_train = np.load("C:\\Users\\ajayg\\Desktop\\University\\Thesis\\Datasets\\example_nondeterminist_cons_train.npy")
spotmarket_data = pd.read_excel("C:\\Users\\ajayg\\Desktop\\University\\Thesis\\Datasets\\spotmarket_data_2007-2013.xls")


# Convert 'Date' column to datetime index in spotmarket_data
spotmarket_data['Date'] = pd.to_datetime(spotmarket_data['Date'])
spotmarket_data.set_index('Date', inplace=True)

# Resample the spotmarket_data to hourly timesteps using forward fill ('ffill')
spotmarket_data = spotmarket_data['BASE (00-24)'].resample('1H').ffill()
consumption_data = nondet_cons_train
production_data = pv_prod_train


# Determine the indices for summer and winter periods
# Assuming the data starts from January 1 and each day has 24 data points
summer_start = 31*24 + 28*24 + 31*24 + 30*24 + 31*24  # Start of June
summer_end = summer_start + 92*24  # End of August
winter_start = 0  # Start of December
winter_end = 31*24 + 31*24 + 28*24  # End of February

# Select summer and winter data
summer_consumption = consumption_data[summer_start:summer_end]
winter_consumption = consumption_data[winter_start:winter_end]
summer_production = production_data[summer_start:summer_end]
winter_production = production_data[winter_start:winter_end]

# Create summer and winter environments
summer_env = MicrogridEnv(summer_consumption, summer_production, spotmarket_data, n_past=5)
winter_env = MicrogridEnv(winter_consumption, winter_production, spotmarket_data, n_past=5)

# Create the environment for the base case
env = MicrogridEnv(consumption_data, production_data, spotmarket_data, n_past=5)

# Run a simple random policy for 10 episodes
for episode in range(10):
    state = env.reset()
    done = False
    score = 0

    while not done:
        # Select a random action
        action = env.action_space.sample()

        # Apply the action to the environment
        next_state, reward, done, _ = env.step(action)  # Capturing the 4 values returned by env.step()

        # Add the reward for this step to the total score for the current episode
        score += reward

    # Print the score for this episode
    print(f'Base Case, Episode: {episode + 1}, Score: {score}')
    
import numpy as np
import torch
import torch.nn as nn


class LSTMPolicy(nn.Module):
    def __init__(self, state_size, action_size, n_past, hidden_size=64):
        super(LSTMPolicy, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(state_size + 2*n_past, hidden_size)  # Modified to accept state_size + n_past
        self.fc = nn.Linear(hidden_size, action_size)
        self.n_past = n_past  # Add this line to store n_past as an instance variable

    def forward(self, x):
        state, past_state = x[:, :-self.n_past, :], x[:, -self.n_past:, :]  # Use self.n_past here
        x, _ = self.lstm(torch.cat((state, past_state), dim=1))  # Concatenate along the sequence dimension
        x = self.fc(x[:, -1, :])
        return x

    
def create_lstm_model(state_size, action_size, n_past, hidden_size=64):
    return LSTMPolicy(state_size, action_size, n_past, hidden_size)

# Define state_size, action_size, and n_past
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
n_past = 5  # or any other value you prefer

# Create the policies and environments for summer and winter
policy_summer = LSTMPolicy(state_size, action_size, n_past)
env_summer = MicrogridEnv(summer_consumption, summer_production, spotmarket_data, n_past)

policy_winter = LSTMPolicy(state_size, action_size, n_past)
env_winter = MicrogridEnv(winter_consumption, winter_production, spotmarket_data, n_past)




import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    def __init__(self, state_size, action_size, replay_memory_size, batch_size, gamma, lr, update_every, n_past):
        print("Initializing DQNAgent")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(replay_memory_size)  # create ReplayMemory instance
        self.batch_size = batch_size
        self.gamma = gamma  # discount factor
        self.lr = lr  # learning rate
        self.update_every = update_every  # how often to update the network
        self.n_past = n_past  # Set the desired number of past time steps
        self.policy = LSTMPolicy(state_size, action_size, n_past)  # create LSTM model instance (Step 3)
        self.steps_done = 0  # for updating the network

    def step(self, state, action, next_state, reward, optimizer):
        print("Stepping through DQNAgent")
        self.memory.push(state, action, next_state, reward)  # save transition to replay memory
        self.steps_done += 1
        if self.steps_done % self.update_every == 0:  # update the network every specified steps
            self.update(optimizer)

    def act(self, state, epsilon=0.0):
        print("Acting with DQNAgent")
        if torch.rand(1) > epsilon:  # exploit
            with torch.no_grad():
                # Reshape the state tensor for LSTM input
                state = torch.tensor(state, dtype=torch.float32).view(1, 1, self.state_size)
                return torch.argmax(self.policy(state)).item()
        else:  # explore
            return np.random.randint(self.action_size)
        
    def update(self, optimizer):
        print("Updating DQNAgent")
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch.next_state if s is not None])

        # Extract past states separately
        past_state_batch = torch.cat([torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch.state], dim=0)
        past_state_batch = past_state_batch[:, -self.n_past:]  # Extract only the past states

        # Concatenate past states for each sample in the batch along the sequence dimension
        past_state_batch = past_state_batch.reshape(self.batch_size * self.n_past, -1)

        # Pack the past states as a sequence
        packed_past_state_batch = rnn_utils.pack_padded_sequence(past_state_batch, lengths=[self.n_past] * self.batch_size, batch_first=True, enforce_sorted=False)

        # Concatenate the current state with the packed past states
        state_batch = torch.cat([torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch.state], dim=0)
        state_batch = state_batch[:, - 1]  # Extract only the current state
        # Assume state_batch is a tensor of shape (batch_size, seq_len, num_features)
        state_batch = torch.randn(batch_size, seq_len, num_features)

        # Split the state into current state and past states
        current_state = state_batch[:, -1, :]
        past_states = state_batch[:, :-1, :]

        # Pack the past states
        packed_past_states = rnn_utils.pack_padded_sequence(past_states, lengths=[n_past]*batch_size, batch_first=True)

        # Pass the packed sequence through the LSTM
        packed_lstm_output, _ = self.policy.lstm(packed_past_states)

        # Pad the packed sequence
        lstm_output, _ = rnn_utils.pad_packed_sequence(packed_lstm_output, batch_first=True)

        # Now lstm_output should be a tensor of shape (batch_size, seq_len, hidden_size)


        # Concatenate the current state with the unpacked past states
        state_batch = torch.cat([state_batch, packed_past_state_batch], dim=1)


        # Reshape the state tensor to match the LSTM input
        state_batch = state_batch.view(self.batch_size, self.n_past + 1, -1)

        # Transpose the sequence and batch dimensions
        state_batch = state_batch.transpose(0, 1)

        # Pad the sequence
        packed_state_batch = rnn_utils.pack_padded_sequence(state_batch, lengths=[self.n_past + 1] * self.batch_size, batch_first=False, enforce_sorted=False)

        # Get the LSTM output using the packed sequence
        packed_lstm_output, _ = self.policy.lstm(packed_state_batch)

        # Pad the packed LSTM output to obtain the final output
        lstm_output, _ = rnn_utils.pad_packed_sequence(packed_lstm_output, batch_first=True)

        # Use the last output of the LSTM as the state-action values
        state_action_values = self.policy.fc(lstm_output[:, -1, :])

        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.policy(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




        

# Define the ReplayMemory class
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)  
    

# Creating DQN agent
state_size = env.observation_space.shape[0] + n_past  # use n_past directly
action_size = env.action_space.n

# Set hyperparameters for the DQNAgent
replay_memory_size = 10000
batch_size = 32
gamma = 0.99
learning_rate = 0.001
update_every = 4
n_past = 5  


# Create the DQNAgent instance
agent = DQNAgent(state_size, action_size, replay_memory_size, batch_size, gamma, learning_rate, update_every, n_past)


# Training loop
n_episodes = 1000
max_t = 1000
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

eps = eps_start

# Create the optimizer
optimizer = optim.Adam(agent.policy.parameters(), lr=learning_rate)  # Use the defined learning_rate

for i_episode in range(1, n_episodes + 1):
    print(f"Starting episode {i_episode}")
    state = env.reset()
    score = 0
    for t in range(max_t):
        print(f"Starting timestep {t}")
        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, next_state, reward, optimizer)
        state = next_state
        score += reward
        if done:
            break
    eps = max(eps_end, eps_decay * eps)  # decrease epsilon
    print(f"Episode: {i_episode}, Score: {score}")
    
    
# Save the model
torch.save(agent.policy.state_dict(),r'C:\Users\ajayg\Desktop\University\Thesis')

# Load the model
agent.policy.load_state_dict(torch.load(r'C:\Users\ajayg\Desktop\University\Thesis'))
agent.policy.eval()


# Evaluation
total_rewards = []
n_episodes = 100

for i_episode in range(1, n_episodes+1):
    print(f"Starting evaluation episode {i_episode}")
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state, 0)  # no exploration
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    total_rewards.append(total_reward)
    print(f"Episode: {i_episode}, Total reward: {total_reward}")

print(f"Average reward over {n_episodes} episodes: {np.mean(total_rewards)}")

import matplotlib.pyplot as plt

def generate_graph(policy, env):
    state = env.reset()
    done = False
    states = []
    actions = []
    while not done:
        action = policy.predict(state)
        state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
    # Generate the graph
    plt.figure()
    plt.plot(states)
    plt.title('State over time')
    plt.show()
    plt.figure()
    plt.plot(actions)
    plt.title('Action over time')
    plt.show()

# Generate the graphs
generate_graph(policy_summer, env_summer)
generate_graph(policy_winter, env_winter)


# In[ ]:




