from typing import List, Callable, Tuple, NoReturn, Any

from gym import Env
from Agents import CausalAgent, DiscreteSwitchboardAgent, ContinuousSwitchboardAgent,\
    get_switchboard_causal_graph, get_almost_right_switchboard_causal_graph, get_blank_switchboard_causal_graph
import copy
import numpy as np
from scm import StructuralCausalModel, BoolSCMGenerator
import random
from episode_evals import EvalFunc, EachStepGoalCheck, FixedLengthEpisode, TwoPhaseFixedEpisode


class Switchboard(Env):
    Agent: DiscreteSwitchboardAgent
    Function = Callable[[], bool]
    Lights: List[bool]

    def __init__(self, agent: CausalAgent,
                 eval_func: EvalFunc,
                 scm: StructuralCausalModel = BoolSCMGenerator.make_switchboard_scm_without_context(),
                 fixed_episode_length: bool = False):
        super(Switchboard, self).__init__()
        self.metrics = {'ep_lengths': [],
                        'rewards': []}
        self.fixed_episode_length = fixed_episode_length
        self.eval_func = eval_func

        # initialize causal model
        self.SCM = scm

        self.lights = [False]*5  # all lights are off

        assert type(agent) == DiscreteSwitchboardAgent or type(agent) == ContinuousSwitchboardAgent, \
            'Wrong agent for this environment'

        self.agent = agent
        self.action_space = self.agent.action_space
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
        self.agent.current_action = self.agent.get_action_from_actionspace_sample(action)

        # apply action
        interv_scm = copy.deepcopy(self.SCM)
        action_successful = False
        if self.agent.current_action[0] == 0:  # intervention action
            interv_scm.do_interventions([('X'+str(self.agent.current_action[1]), lambda: self.agent.current_action[2])])
            action_successful = True
        elif self.agent.current_action[0] == 1:
            action_successful = self.agent.update_model_per_action(self.agent.current_action)
        elif self.agent.current_action[0] == None or self.agent.current_action[0] == -1:
            action_successful = True

        self.steps_this_episode += 1

        # determine the states of the lights according to the causal structure
        self.lights = interv_scm.get_next_instantiation()[0]

        self.agent.store_observation_per_action(self.lights)

        # reverse all wrong edges, this could eventually speed up learning
        #self.agent.reverse_wrong_edges(0.1)

        # determine state after action
        self.last_observation = self.update_get_obs_vector()

        # evaluate the step
        done, reward = self.eval_func.evaluate_step(action_successful)

        self.prev_action = self.agent.current_action
        self.metrics['rewards'].append(reward)
        if type(self.agent) == ContinuousSwitchboardAgent:
            print([round(a) for a in self.agent.current_action], '\treward', reward)
        else:
            if self.fixed_episode_length and done:
                print('episode reward', reward)
            elif self.fixed_episode_length and not done:
                pass
            else:
                print(self.agent.current_action, '\treward', reward)

        # reset environment if episode is done
        if done:
            self.reset()

        return self.last_observation, reward, done, {}

    def update_get_obs_vector(self) -> np.ndarray:
        # push old observations
        for i in range(1, len(self.old_obs)):
            self.old_obs[i-1] = self.old_obs[i]
            
        intervention_one_hot = [1.0 if self.agent.current_action[1] == i else 0.0 for i in range(len(self.lights))]
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
                if self.agent.current_action[1] == i:
                    out += '*'
                out += '\t'
            print(out)


class ReservoirSwitchboard(Switchboard):
    """Same as Switchboard only that a new scm is loaded after each episode"""
    def __init__(self, reservoir: List[StructuralCausalModel],
                 agent: CausalAgent,
                 fixed_episode_length: bool = False):
        super(ReservoirSwitchboard, self).__init__(agent, scm=reservoir[0], fixed_episode_length=fixed_episode_length)
        self.scm_reservoir = reservoir

    def reset(self):
        self.SCM = random.sample(self.scm_reservoir, 1)[0]
        return super(ReservoirSwitchboard, self).reset()

