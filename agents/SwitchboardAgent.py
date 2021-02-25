from typing import Tuple, List, Optional, NoReturn
from causalnex.structure import StructureModel
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


class SwitchboardAgent:
    current_mode: str
    causal_model: StructureModel
    action = Tuple[Optional[int], Optional[bool]]
    actions: List[action]
    current_action: action

    def __init__(self, n_switches):
        # create a dictonary of actions that can be performed on the switchboard
        self.actions = [(i, True) for i in range(n_switches)]
        self.actions.extend([(i, False) for i in range(n_switches)])
        self.actions.append((None, None))
        self.current_action = (None, None)
        self.var_names = ['x'+str(i) for i in range(n_switches)]

        # initialize causal model
        self.causal_model = StructureModel()
        [self.causal_model.add_node(name) for name in self.var_names]
        # TODO: what about edges?

        # initialize the storages for observational and interventional data. (None, None) is the purely observational data
        self.collected_data = {str(action): pd.DataFrame(columns=self.var_names) for action in self.actions}
    
    def store_observation(self, obs: List[bool]):
        # put the observation in the right dataframe
        obs_dict = {self.var_names[i]: obs[i] for i in range(len(self.var_names))}
        self.collected_data[str(self.current_action)] = self.collected_data[str(self.current_action)].append(obs_dict, ignore_index=True)

        # Update probability table
        
    def display_causal_model(self) -> NoReturn:
        fig, ax = plt.subplots()
        nx.draw_circular(self.causal_model, ax=ax, with_labels=True)
        fig.show()

