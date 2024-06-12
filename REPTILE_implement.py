import torch
from torch import nn, optim
import numpy as np
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BaseWrapper, BlueFlatWrapper

seed = 0
innerstepsize = 0.02  # stepsize in inner SGD
innerepochs = 1  # number of epochs of each inner SGD
outerstepsize0 = 0.1  # stepsize of outer optimization, i.e., meta-optimization
niterations = 30000  # number of outer updates

# CAGE simulation step
num_steps = 500

np.random.seed(seed)
torch.manual_seed(seed)

# Define task distribution
def gen_task():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=num_steps
    )
    cyborg = CybORG(scenario_generator=sg)
    return BlueFlatWrapper(cyborg,pad_spaces = True)

# Define model
class BlueAgentModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BlueAgentModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

def test_on_task(env):
    obs, info = env.reset()
    total_loss = 0
    for step in range(num_steps):
        actions={}
        for agent_name in obs.keys():
            last_obs = obs[agent_name]
            last_mask = info[agent_name]['action_mask']
            obs_tensor = torch.tensor(last_obs, dtype=torch.float32).unsqueeze(0)
            action_logits = model(obs_tensor)
            for i in range(len(last_mask)):
                if not last_mask[i]:
                    action_logits[0][i] = 0
            action = torch.argmax(action_logits, dim=1).item()
            actions[agent_name] = action
        obs, rewards, terminated_set,truncated_set, info = env.step(actions=actions)
        for agent_name, reward in rewards.items():
            total_loss-=reward
    return total_loss

def train_on_task(env):
    obs, info = env.reset()
    total_loss = 0

    for step in range(num_steps):
        actions={}
        for agent_name in obs.keys():
            last_obs = obs[agent_name]
            last_mask = info[agent_name]['action_mask']
            obs_tensor = torch.tensor(last_obs, dtype=torch.float32).unsqueeze(0)
            action_logits = model(obs_tensor)
            for i in range(len(last_mask)):
                if not last_mask[i]:
                    action_logits[0][i] = 0
            action = torch.argmax(action_logits, dim=1).item()
            actions[agent_name] = action


        obs, rewards, terminated_set,truncated_set, info = env.step(actions=actions)

        losses = []
        for agent_name, reward in rewards.items():
            reward_tensor = torch.tensor(float(reward), requires_grad=True)
            loss = -reward_tensor
            losses.append(loss)

        total_loss = sum(losses)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    return total_loss

env = gen_task()
obs_space = env.observation_space('blue_agent_4').shape[0]
act_space = env.action_space('blue_agent_4').n

model = BlueAgentModel(obs_space, 64, act_space)
optimizer = optim.Adam(model.parameters(), lr=innerstepsize)

baseline=gen_task()

# Reptile training loop
for iteration in range(niterations):
    weights_before = deepcopy(model.state_dict())

    # Generate task
    env = gen_task()

    # Do SGD on this task
    train_on_task(env)

    # Interpolate between current weights and trained weights from this task
    weights_after = model.state_dict()
    outerstepsize = outerstepsize0 * (1 - iteration / niterations)  # linear schedule
    model.load_state_dict({
        name: weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize
        for name in weights_before
    })
    print(f"Iteration {iteration + 1}/{niterations} complete.")
    if (iteration + 1) % 10 == 0:
        print('reward: ',test_on_task(baseline))
        torch.save(model.state_dict(), f'model_{iteration}.pth')

print("Training complete.")
