from typing import Tuple, List, Optional, NoReturn
from causalnex.structure import StructureModel
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pandas import DataFrame


class SwitchboardAgent:
    collected_data: dict
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

    def get_est_postint_distrib(self, query: str, do: action) -> DataFrame:
        '''Computes and returns P(Query | do(action))'''
        query_dataframe = self.collected_data[str(do)]
        query_dataframe = query_dataframe.groupby(query).size()/len(query_dataframe)
        return query_dataframe.rename('P('+query+'|do'+str(do)+')')

    def get_est_avg_causal_effect(self, query: str, action1: action, action2: action) -> float:
        assert action1[0] == action2[0], 'effect can only be measured on the same intervention variable'

        dist1 = self.get_est_postint_distrib(query, action1)
        dist2 = self.get_est_postint_distrib(query, action2)

        #change boolean index to 0 and 1s
        if type(dist1.index[0] == bool):
            dist1 = dist1.rename({True: 1, False: 0})
            dist2 = dist2.rename({True: 1, False: 0})

        exp_val1 = sum(dist1.index.values * dist1._values)
        exp_val2 = sum(dist2.index.values * dist2._values)

        return exp_val1 - exp_val2

    def get_est_cond_distr(self, query: str, condition: Tuple[str, bool]) -> DataFrame:
        obs_data = self.collected_data['(None, None)']
        obs_data = obs_data[obs_data[condition[0]] == condition[1]]
        obs_data = obs_data.groupby(query).size() / len(obs_data)
        return obs_data.rename('P('+query+'|'+condition[0]+'='+str(condition[1])+')')

    def display_causal_model(self) -> NoReturn:
        fig, ax = plt.subplots()
        nx.draw_circular(self.causal_model, ax=ax, with_labels=True)
        fig.show()

