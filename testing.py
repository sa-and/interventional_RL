from Agents import ContinuousSwitchboardAgent, DiscreteSwitchboardAgent, get_almost_right_switchboard_causal_graph
from Environments import BoolSCMGenerator, SwitchboardReservoir, Switchboard
from episode_evals import FixedLengthEpisode, NoEval
import cdt
from cdt.causality.graph import PC, LiNGAM, GES
from pandas import DataFrame, Series
import networkx as nx
import matplotlib.pyplot as plt
from stable_baselines import ACER
import gym
import random
from Environments import Dumb
from stable_baselines.common.policies import MlpLstmPolicy

model = ACER(MlpLstmPolicy, Dumb(),
                     policy_kwargs={'net_arch': [10,
                                                 'lstm',
                                                 {'pi': [10],
                                                  'vf': [10]}],
                                    'n_lstm': 10},

                     n_steps=10,
                     n_cpu_tf_sess=8,
                     replay_ratio=5,
                     buffer_size=500000,
                     verbose=1
                     )

model.learn(50000)
