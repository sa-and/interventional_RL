from typing import List
from Environments import Switchboard, ReservoirSwitchboard
from Agents import DiscreteSwitchboardAgent, ContinuousSwitchboardAgent
from stable_baselines.a2c import A2C
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.deepq.policies import MlpPolicy as dqnMlpPolicy
from stable_baselines import DQN, DDPG, ACER
import random
import stable_baselines.common.vec_env as venv
import pickle
from scm import StructuralCausalModel, BoolSCMGenerator
from tqdm import tqdm
from episode_evals import FixedLengthEpisode, EachStepGoalCheck



#
#
# def create_switchboard_a2c_fixed():
#     agent = DiscreteSwitchboardAgent(5)
#     a = Switchboard(agent, fixed_episode_length=True)
#     return a
#
#
# def create_switchboard_a2c_dynamic():
#     agent = DiscreteSwitchboardAgent(5)
#     a = Switchboard(agent, fixed_episode_length=False)
#     return a
#
#
# def train_switchboard_acer(steps: int, workers: int = 8, fixed_length: bool = False):
#     if fixed_length:
#         switchboard = venv.DummyVecEnv([create_switchboard_a2c_fixed for i in range(workers)])
#     else:
#         switchboard = venv.DummyVecEnv([create_switchboard_a2c_dynamic for i in range(workers)])
#
#     # data collection phase
#     for i in range(500):
#         a = switchboard.envs[0].action_space.sample()
#         a = [a for i in range(workers)]
#         switchboard.step(a)
#     print('data collection phase done\n\n\n\n\n\n\n\n\n\n')
#
#     model = ACER(MlpLstmPolicy, switchboard,
#                  policy_kwargs={'net_arch': [30,
#                                              'lstm',
#                                              {'pi': [50],
#                                               'vf': [10]}],
#                                 'n_lstm': 100},
#
#                  n_steps=50,
#                  n_cpu_tf_sess=8,
#                  replay_ratio=5,
#                  buffer_size=500000
#                  )
#     model = ACER.load('experiments/preliminary/exp19/model.zip', switchboard)
#     model.learn(steps)
#     title = 'ACER, discrete agnt, fixed = ' + str(fixed_length)
#     plt.title(title)
#     plt.plot(switchboard.envs[0].metrics['rewards'])
#     plt.show()
#     return model, switchboard
#
#
# def train_switchboard_a2c(steps: int, workers: int = 8, fixed_length: bool = False):
#     if fixed_length:
#         switchboard = venv.DummyVecEnv([create_switchboard_a2c_fixed for i in range(workers)])
#     else:
#         switchboard = venv.DummyVecEnv([create_switchboard_a2c_dynamic for i in range(workers)])
#
#     # data collection phase
#     for i in range(500):
#         a = switchboard.envs[0].action_space.sample()
#         a = [a for i in range(workers)]
#         switchboard.step(a)
#     print('data collection phase done\n\n\n\n\n\n\n\n\n\n')
#
#     model = A2C(MlpLstmPolicy, switchboard,
#                 learning_rate=0.0001,
#                 policy_kwargs={'net_arch': [50,
#                                             'lstm',
#                                             {'pi': [50],
#                                              'vf': [10]}],
#                                'n_lstm': 20},
#                 epsilon=0.05,
#                 n_steps=5,
#                 n_cpu_tf_sess=8)
#
#     model.learn(steps)
#     title = 'A2C, cont agnt, fixed = ' + str(fixed_length)
#     plt.title(title)
#     plt.plot(switchboard.envs[0].metrics['rewards'])
#     plt.show()
#     return model, switchboard
#
#
# def train_switchboard_dqn(steps: int, fixed_length):
#     agent = DiscreteSwitchboardAgent(5, state_repeats=3, causal_graph=get_blank_switchboard_causal_graph())
#     switchboard = Switchboard(agent, fixed_episode_length=fixed_length)
#
#     # data collection phase
#     for i in range(1000):
#         switchboard.step(-1)
#     data_actions = [i for i in range(10)]
#     for i in range(500):
#         a = random.sample(data_actions, k=1)[0]
#         switchboard.step(a)
#
#     model = DQN(dqnMlpPolicy, switchboard,
#                 buffer_size=200000,
#                 learning_rate=0.001,
#                 policy_kwargs={'layers': [80, 60]},
#                 exploration_final_eps=0.05,
#                 batch_size=64,
#                 n_cpu_tf_sess=8)
#     #model = DQN.load('models/exp10.zip', switchboard)
#     model.learn(steps)
#
#     title = 'DQN, fixed = ' + str(fixed_length)
#     plt.title(title)
#     plt.plot(switchboard.metrics['rewards'])
#     plt.show()
#     return model, switchboard
#
#
# def train_switchboard_ddpg(steps: int):
#     agent = ContinuousSwitchboardAgent(5, state_repeats=3)
#     switchboard = Switchboard(agent)
#
#     n_actions = switchboard.action_space.shape[-1]
#     param_noise = None
#     action_noise = NormalActionNoise(0, 0.1)
#
#     model = DDPG(ddpgMlpPolicy,
#                  switchboard,
#                  param_noise=param_noise,
#                  action_noise=action_noise,
#                  policy_kwargs={'layers': [90, 40, 10]},
#                  n_cpu_tf_sess=8,
#                  buffer_size=50000)
#     model.learn(steps)
#
#     plt.title('ddpg')
#     plt.plot(switchboard.metrics['rewards'])
#     plt.show()
#     return model, switchboard

# check = check_env(swtchbrd)

def load_dataset(path):
    with open(path, 'rb') as f:
        dic = pickle.load(f)
    return dic


# For multiprocessing
def make_res_switchboard_constructor(scms: List[StructuralCausalModel],
                                     n_switches,
                                     fixed_length: bool = True,
                                     discrete_agent: bool = True):
    def make_switchboard_env():
        if discrete_agent:
            agent = DiscreteSwitchboardAgent(n_switches)
        else:
            agent = ContinuousSwitchboardAgent(n_switches)
        switchboard = ReservoirSwitchboard(agent=agent, reservoir=scms, fixed_episode_length=fixed_length)
        return switchboard

    return make_switchboard_env


def make_switchboard_constructor(scm: StructuralCausalModel,
                                 n_switches: int,
                                 fixed_length: bool = True,
                                 discrete_agent: bool = True,
                                 seed: int = 0):
    def make_switchboard_env():
        if discrete_agent:
            agent = DiscreteSwitchboardAgent(n_switches)
        else:
            agent = ContinuousSwitchboardAgent(n_switches)
        eval_func = EachStepGoalCheck(agent, 0.15, )
        switchboard = Switchboard(agent=agent,
                                  scm=scm,
                                  fixed_episode_length=fixed_length,
                                  eval_func=eval_func)
        switchboard.seed(seed)
        return switchboard

    return make_switchboard_env


def train_switchboard_acer(steps: int,
                           train_scms: List[StructuralCausalModel],
                           fixed_length: bool = True,
                           discrete_agent: bool = True,
                           load_model_path: str = None,
                           workers: int = 1):
    # possibly set up multiprocessing environments
    if len(train_scms) == 1:
        # create all workers with the same environment
        switchboard = venv.SubprocVecEnv([make_switchboard_constructor(train_scms[0], 5, fixed_length, discrete_agent, w)
                                         for w in range(workers)],
                                         start_method='spawn')

    elif len(train_scms) <= workers:
        # start each worker with a different environment
        switchboard = venv.SubprocVecEnv([make_switchboard_constructor(train_scms[i], 5, fixed_length, discrete_agent, i)
                                          for i in range(len(train_scms))],
                                         start_method='spawn')

    if len(train_scms) > workers:
        # no multiprocessing. For possible implementation see below
        switchboard = venv.DummyVecEnv([make_switchboard_constructor(train_scms[i], 5, fixed_length, discrete_agent)
                                        for i in range(len(train_scms))])

        # this part needs to be implemented properly for multiprocessing
        # create all training environments in _workers_ parallel processes
        # stepsize = int(len(train_scms) / workers)
        # switchboard = venv.SubprocVecEnv([make_res_switchboard_constructor(scms=train_scms[w:(w + stepsize)],
        #                                                                    n_switches=5,
        #                                                                    fixed_length=fixed_length,
        #                                                                    discrete_agent=discrete_agent)
        #                                   for w in range(workers)],
        #                                  start_method='spawn')

    # data collection phase in order to approximate the distribution correctly
    for i in tqdm(range(3000)):
        a = [switchboard.action_space.sample() for i in range(switchboard.num_envs)]#[[switchboard.envs[i].envs[j].action_space.sample() for j in range(len(switchboard.envs[i].num_envs))] for i in range(switchboard.num_envs)]
        switchboard.step(a)
    print('data collection phase done\n\n\n\n\n\n\n\n\n\n')

    # load pretrained model
    if load_model_path:
        model = ACER.load(load_model_path, switchboard)

    # Create new model
    else:
        model = ACER(MlpLstmPolicy, switchboard,
                     policy_kwargs={'net_arch': [40,
                                                 'lstm',
                                                 {'pi': [80],
                                                  'vf': [10]}],
                                    'n_lstm': 100},

                     n_steps=50,
                     n_cpu_tf_sess=8,
                     replay_ratio=10,
                     buffer_size=500000
                     )

    model.learn(steps)

    # plot rewards of first environment for debugging
    #title = 'ACER, discrete agnt, fixed = ' + str(fixed_length)
    #plt.title(title)
    #plt.plot(switchboard.envs[0].metrics['rewards'])
    #plt.show()

    return model, switchboard


if __name__ == '__main__':
    model_save_path = 'experiments/actual/exptest'

    # load train and test set
    scms = load_dataset('data/scms/switchboard/5x5var_all.pkl')
    scms_train = [scms[3], scms[119]]
    scms_train = [BoolSCMGenerator.make_switchboard_scm_with_context()]
    model, board = train_switchboard_acer(2000000,
                                          train_scms=scms_train,
                                          fixed_length=True,
                                          discrete_agent=True,
                                          workers=6,
                                          load_model_path=None)

    model.save(model_save_path + 'model')
    # with open(model_save_path + 'metrics.pkl', 'wb') as f:
    #     pickle.dump(board.envs[0].metrics, f, pickle.HIGHEST_PROTOCOL)
# with open('experiments/preliminary/exp15/metrics.pkl', 'rb') as f:
#     dic = pickle.load(f)
#
# print(dic)
#model = DQN.load('models/exp3.zip', swtchbrd)

