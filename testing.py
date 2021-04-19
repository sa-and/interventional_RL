from Agents import ContinuousSwitchboardAgent, DiscreteSwitchboardAgent, get_almost_right_switchboard_causal_graph
from Environments import BoolSCMGenerator, SwitchboardReservoir, Switchboard
from episode_evals import FixedLengthEpisode, NoEval
import cdt
from cdt.causality.graph import PC, LiNGAM, GES
from pandas import DataFrame, Series
import networkx as nx
import matplotlib.pyplot as plt
cdt.SETTINGS.rpath = 'C:/Program Files/R/R-4.0.5/bin/Rscript'

scm = BoolSCMGenerator.load_dataset('data/scms/switchboard/3x3var_all.pkl')[9]
# agent = DiscreteSwitchboardAgent(3)
# env = Switchboard(agent, NoEval(), scm=scms[19])
columns = ['X0', 'X1', 'X2']
obs_data = DataFrame(columns=columns)
lights = []
for i in range(1000):
    lights = scm.get_next_instantiation()[0]
    dic = {'X'+str(i): float(lights[i]) for i in range(len(lights))}
    obs_data = obs_data.append({'X'+str(i): float(lights[i]) for i in range(len(lights))}, ignore_index=True)

print(obs_data)


algo = GES()
graph = algo.predict(obs_data)
nx.draw_networkx(graph)
plt.show()
nx.draw_networkx(BoolSCMGenerator.create_graph_from_scm(scm))
plt.show()
print()