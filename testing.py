from Environments import Switchboard, StructuralCausalModel
from stable_baselines.a2c import A2C
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.env_checker import check_env
import matplotlib.pyplot as plt
import random
import stable_baselines.common.vec_env as venv

from Agents import SwitchboardAgentA2C
agent = SwitchboardAgentA2C(5)
env = Switchboard(agent)
model = A2C(MlpLstmPolicy, env,
            learning_rate=0.001,
            policy_kwargs={'net_arch': [24,
                                        30,
                                        'lstm',
                                        {'pi': [45],
                                         'vf': [10]}],
                           'n_lstm': 30},
            epsilon=0.005)
model.learn(1000)