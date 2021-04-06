from typing import List, Callable, Tuple, NoReturn, Any

from gym import Env
from Agents import CausalAgent, DiscreteSwitchboardAgent, ContinuousSwitchboardAgent,\
    get_switchboard_causal_graph, get_almost_right_switchboard_causal_graph, get_blank_switchboard_causal_graph
import copy
import numpy as np
from scm import StructuralCausalModel, BoolSCMGenerator


class Switchboard(Env):
    Agent: DiscreteSwitchboardAgent
    Function = Callable[[], bool]
    Lights: List[bool]

    def __init__(self, agent: CausalAgent, scm: StructuralCausalModel = None, fixed_episode_length: bool = False):
        super(Switchboard, self).__init__()
        self.metrics = {'ep_lengths': [],
                        'rewards': []}
        self.fixed_episode_length = fixed_episode_length

        # initialize causal model
        if scm == None:
            self.SCM = BoolSCMGenerator.make_switchboard_scm_without_context()
        else:
            self.SCM = scm

        self.lights = [False]*5  # all lights are off

        assert type(agent) == DiscreteSwitchboardAgent or type(agent) == ContinuousSwitchboardAgent, \
            'Wrong agent for this environment'

        self.agent = agent
        self.action_space = self.agent.action_space
        if type(self.agent) == DiscreteSwitchboardAgent:
            self.current_action = (None, None, None)
        elif type(self.agent) == ContinuousSwitchboardAgent:
            self.current_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.observation_space = self.agent.observation_space
        self.prev_action = None
        self.last_observation = None
        self.old_obs = []
        for i in range(self.agent.state_repeats):
            self.old_obs.append([0.0 for i in range(int(self.observation_space.shape[0]/self.agent.state_repeats))])

        self.steps_this_episode = 0
        self.observation = self.update_get_obs_vector()

    def reset(self) -> np.ndarray:
        self.steps_this_episode = 0
        # self.agent.set_causal_model(get_blank_switchboard_causal_graph())
        self.agent.random_reset_causal_model()
        # reset observations
        self.old_obs = []
        for i in range(self.agent.state_repeats):
            self.old_obs.append([0.0 for i in range(int(self.observation_space.shape[0] / self.agent.state_repeats))])

        return self.update_get_obs_vector()

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        self.current_action = self.agent.get_action_from_actionspace_sample(action)

        # apply action
        interv_scm = copy.deepcopy(self.SCM)
        action_successful = False
        if self.current_action[0] == 0:  # intervention action
            interv_scm.do_interventions([('X'+str(self.current_action[1]), lambda: self.current_action[2])])
            action_successful = True
        elif self.current_action[0] == 1:
            action_successful = self.agent.update_model_per_action(self.current_action)
        elif self.current_action[0] == None or self.current_action[0] == -1:
            action_successful = True

        self.steps_this_episode += 1

        # determine the states of the lights according to the causal structure
        self.lights = interv_scm.get_next_instantiation()[0]

        self.agent.store_observation_per_action(self.lights, self.current_action)

        # reverse all wrong edges, this could eventually speed up learning
        #self.agent.reverse_wrong_edges(0.1)

        # determine state after action
        self.last_observation = self.update_get_obs_vector()

        if self.fixed_episode_length:
            done, reward = self.do_fixed_eval(action_successful, 50)
        else:
            # let the episode end when the causal model is fully learned (loss reaching threshold of -0.006)
            done, reward = self.do_flexible_eval(action_successful)

        self.prev_action = self.current_action
        self.metrics['rewards'].append(reward)
        if type(self.agent) == ContinuousSwitchboardAgent:
            print([round(a) for a in self.current_action], '\treward', reward)
        else:
            if self.fixed_episode_length and done:
                print('episode reward', reward)
            elif self.fixed_episode_length and not done:
                pass
            else:
                print(self.current_action, '\treward', reward)

        return self.last_observation, reward, done, {}

    def do_fixed_eval(self, action_successful: bool,
                      length_per_episode: int,
                      allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        '''
        Ends the episode after 'length_per_episode' steps. Here we only give a reward for the
        achievement of the goal at the end of each episode. A negative reward for illegal actions
        is still returned.
        :param action_successful:
        :param length_per_episode:
        :param allow_unsuccessful_actions:
        :return:
        '''
        done = almost_done = very_almost_done = learned = False
        if self.steps_this_episode >= length_per_episode:
            done = True
            n_wrong_edges = self.agent.has_wrong_edges(0.1)
            n_missing_edges = self.agent.has_missing_edges(0.1)
            learned = (n_wrong_edges + n_missing_edges == 0)
            almost_done = (n_wrong_edges + n_missing_edges < 4) and (n_wrong_edges + n_missing_edges >= 2)
            very_almost_done = (n_wrong_edges + n_missing_edges < 2) and (n_wrong_edges + n_missing_edges > 0)

        if not action_successful and not allow_unsuccessful_actions:  # illegal action was taken
            reward = -1
        elif almost_done:
            reward = 2
        elif very_almost_done:
            reward = 5
        elif learned:
            reward = 30
            self.agent.display_causal_model()
            self.reset()
        else:
            reward = 0

        return done, reward

    def do_flexible_eval(self, action_successful: bool, allow_unsuccessful_actions: bool = True) -> Tuple[bool, float]:
        '''
        Ends the episode whenever a graph-altering action is performed
        :param action_successful:
        :param allow_unsuccessful_actions:
        :return:
        '''
        if self.current_action[0] == 1:  # only check if the model actually changed.
            n_wrong_edges = self.agent.has_wrong_edges(0.1)
            n_missing_edges = self.agent.has_missing_edges(0.1)
            done = (n_wrong_edges + n_missing_edges == 0)
            almost_done = (n_wrong_edges + n_missing_edges < 4) and (n_wrong_edges + n_missing_edges >= 2)
            very_almost_done = (n_wrong_edges + n_missing_edges < 2) and (n_wrong_edges + n_missing_edges > 0)

        else:
            done = False
            almost_done = False
            very_almost_done = False

        # compute reward
        if not action_successful and not allow_unsuccessful_actions:  # illegal action was taken
            reward = -1
        elif almost_done:
            reward = 2
        elif very_almost_done:
            reward = 5
        elif done:  # the graph has been learned
            reward = 30
            self.metrics['ep_lengths'].append(self.steps_this_episode)
            self.reset()
        else:  # intervention, non-intervention, graph-changing
            reward = 0
        return done, reward

    def update_get_obs_vector(self) -> np.ndarray:
        # push old observations
        for i in range(1, len(self.old_obs)):
            self.old_obs[i-1] = self.old_obs[i]
            
        intervention_one_hot = [1.0 if self.current_action[1] == i else 0.0 for i in range(len(self.lights))]
        graph_state = self.agent.get_graph_state()
        state = [float(l) for l in self.lights]  # convert bool to int
        state.extend(intervention_one_hot)
        state.extend(graph_state)
        self.old_obs[-1] = state
        self.observation = np.array(self.old_obs).flatten()
        return self.observation

    def render(self, mode: str = 'human') -> NoReturn:
        if mode == 'human':
            out = ''
            for i in range(len(self.lights)):
                if self.lights[i]:
                    out += '|'
                else:
                    out += 'O'
                if self.current_action[1] == i:
                    out += '*'
                out += '\t'
            print(out)

