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
from Environments import BoolSCMGenerator, SwitchboardReservoir
from episode_evals import FixedLengthEpisode

env = SwitchboardReservoir(BoolSCMGenerator.make_obs_equ_3var_envs(),
                           3,
                           DiscreteSwitchboardAgent,
                           FixedLengthEpisode)
env.collect_interv_data(1000)
print()