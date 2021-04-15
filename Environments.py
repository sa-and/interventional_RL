from typing import List, Callable, Tuple, NoReturn, Any

from gym import Env
from Agents import CausalAgent, DiscreteSwitchboardAgent, ContinuousSwitchboardAgent,\
    get_switchboard_causal_graph, get_almost_right_switchboard_causal_graph, get_blank_switchboard_causal_graph
import copy
import numpy as np
from scm import StructuralCausalModel, BoolSCMGenerator
import random
from episode_evals import EvalFunc, EachStepGoalCheck, FixedLengthEpisode, TwoPhaseFixedEpisode
from tqdm import tqdm


class Switchboard(Env):
    Agent: DiscreteSwitchboardAgent
    Function = Callable[[], bool]
    Lights: List[bool]

    def __init__(self, agent: CausalAgent,
                 eval_func: EvalFunc,
                 scm: StructuralCausalModel = BoolSCMGenerator.make_switchboard_scm_without_context()):
        super(Switchboard, self).__init__()
        self.metrics = {'ep_lengths': [],
                        'rewards': []}
        self.eval_func = eval_func

        # initialize causal model
        self.SCM = scm

        self.lights = [False]*len(self.SCM.endogenous_vars)  # all lights are off

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
            if type(self.eval_func) == FixedLengthEpisode or type(self.eval_func) == TwoPhaseFixedEpisode:
                if done:
                    print('episode reward', reward)
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


class SwitchboardReservoir(Env):
    envs: List[Switchboard]

    def __init__(self, scms: List[StructuralCausalModel],
                 n_switches: int,
                 agent_type: type(CausalAgent),
                 eval_func_type: type(EvalFunc)):

        self.envs = []
        for scm in scms:
            agent = agent_type(n_switches)
            eval_func = eval_func_type(agent, 0.2, 20)
            self.envs.append(Switchboard(agent, eval_func, scm))

        self.current_env = self.envs[0]

    def reset(self):
        # reset the current environment
        self.current_env.reset()

        # choose a random next environment and reset it
        self.current_env = random.choice(self.envs)
        return self.current_env.reset()

    def step(self, action):
        return self.current_env.step(action)

    def render(self, mode='human'):
        self.current_env.render(mode)

    def collect_interv_data(self, n_collections_per_env: int):
        """
        Performs n_collections_per_env interventional steps in each environment of the reservoir in order
        to approximate the interventional distribution. This should be done every time before training
        :param n_collections_per_env: How many datapoints to collect for each environment
        """
        print('Collecting interventional data...')
        bar = tqdm(total=len(self.envs)*n_collections_per_env)
        for e in self.envs:
            i = 0
            while i < n_collections_per_env:
                action = e.action_space.sample()
                if e.agent.get_action_from_actionspace_sample(action)[0] == 1:
                    pass
                else:
                    e.step(action)
                    i += 1
                    bar.update(1)
