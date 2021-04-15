from Agents import ContinuousSwitchboardAgent, DiscreteSwitchboardAgent, get_almost_right_switchboard_causal_graph
from Environments import BoolSCMGenerator, SwitchboardReservoir
from episode_evals import FixedLengthEpisode


scms = BoolSCMGenerator.load_dataset('data/scms/switchboard/3x3var_all.pkl')
env = SwitchboardReservoir(BoolSCMGenerator.make_obs_equ_3var_envs(),
                           3,
                           DiscreteSwitchboardAgent,
                           FixedLengthEpisode)
env.collect_interv_data(1000)
print()