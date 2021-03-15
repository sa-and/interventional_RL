from Environments import Switchboard, StructuralCausalModel
from stable_baselines.a2c import A2C
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.env_checker import check_env
import matplotlib.pyplot as plt
import random
import stable_baselines.common.vec_env as venv

from Agents import ContinuousSwitchboardAgent, DiscreteSwitchboardAgent, get_almost_right_switchboard_causal_graph


agent = DiscreteSwitchboardAgent(5, get_almost_right_switchboard_causal_graph())
env = Switchboard(agent)

# data collection phase
for i in range(100):
    env.step(-1)
data_actions = [i for i in range(10)]
for i in range(1000):
    a = random.sample(data_actions, k=1)[0]
    env.step(a)

print(agent.has_wrong_edges(0.1))