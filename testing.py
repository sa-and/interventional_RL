from Environments import Switchboard, StructuralCausalModel
from stable_baselines.a2c import A2C
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.env_checker import check_env
import matplotlib.pyplot as plt
import random
import stable_baselines.common.vec_env as venv
import networkx as nx
import dill

from Agents import ContinuousSwitchboardAgent, DiscreteSwitchboardAgent, get_almost_right_switchboard_causal_graph
from Environments import BoolSCMGenerator

agent = DiscreteSwitchboardAgent(5, get_almost_right_switchboard_causal_graph())
env = Switchboard(agent)

# data collection phase
# for i in range(100):
#     env.step(-1)
# data_actions = [i for i in range(10)]
# for i in range(1000):
#     a = random.sample(data_actions, k=1)[0]
#     env.step(a)
#
# print(agent.has_missing_edges(0.1))

gen = BoolSCMGenerator(6, 0)
scms = gen.create_n(200000)
with open('data/scms/switchboard/6x0var_200000.pkl', 'wb') as file:
    dill.dump(scms, file)

print()