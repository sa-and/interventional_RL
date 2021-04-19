from typing import List
from Environments import Switchboard, SwitchboardReservoir
from Agents import DiscreteSwitchboardAgent
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines import ACER
import stable_baselines.common.vec_env as venv
from scm import StructuralCausalModel, BoolSCMGenerator
import copy
from episode_evals import FixedLengthEpisode


# def make_switchboard_constructor(scm: StructuralCausalModel,
#                                  n_switches: int,
#                                  discrete_agent: bool = True,
#                                  seed: int = 0):
#     def make_switchboard_env():
#         if discrete_agent:
#             agent = DiscreteSwitchboardAgent(n_switches)
#         else:
#             agent = ContinuousSwitchboardAgent(n_switches)
#         eval_func = TwoPhaseFixedEpisode(agent, 0.2, 10, 10)
#         switchboard = Switchboard(agent=agent,
#                                   scm=scm,
#                                   eval_func=eval_func)
#         switchboard.seed(seed)
#         return switchboard
#
#     return make_switchboard_env


def train_switchboard_acer(steps: int,
                           train_scms: List[StructuralCausalModel],
                           load_model_path: str = None,
                           workers: int = 1,
                           n_switches: int = 3):

    board = SwitchboardReservoir(train_scms,
                                 n_switches,
                                 DiscreteSwitchboardAgent,
                                 FixedLengthEpisode)

    # data collection phase
    board.collect_interv_data(200)
    print('data collection phase done\n\n\n\n\n\n\n\n\n\n')

    env = venv.SubprocVecEnv([lambda: copy.deepcopy(board) for w in range(workers)], start_method='spawn')
    # # possibly set up multiprocessing environments
    # if len(train_scms) == 1:
    #     # create all workers with the same environment
    #     switchboard = venv.SubprocVecEnv([make_switchboard_constructor(train_scms[0], n_switches, discrete_agent, w)
    #                                      for w in range(workers)],
    #                                      start_method='spawn')
    #
    # elif len(train_scms) <= workers:
    #     # start each worker with a different environment
    #     switchboard = venv.SubprocVecEnv([make_switchboard_constructor(train_scms[i], n_switches, discrete_agent, i)
    #                                       for i in range(len(train_scms))],
    #                                      start_method='spawn')
    #
    # if len(train_scms) > workers:
    #     # no multiprocessing. For possible implementation see below
    #     switchboard = venv.DummyVecEnv([make_switchboard_constructor(train_scms[i], n_switches, discrete_agent)
    #                                     for i in range(len(train_scms))])
    #
    #     # this part needs to be implemented properly for multiprocessing
    #     # create all training environments in _workers_ parallel processes
    #     # stepsize = int(len(train_scms) / workers)
    #     # switchboard = venv.SubprocVecEnv([make_res_switchboard_constructor(scms=train_scms[w:(w + stepsize)],
    #     #                                                                    n_switches=5,
    #     #                                                                    fixed_length=fixed_length,
    #     #                                                                    discrete_agent=discrete_agent)
    #     #                                   for w in range(workers)],
    #     #                                  start_method='spawn')

    # data collection phase in order to approximate the distribution correctly
    # for i in tqdm(range(3000)):
    #     a = [switchboard.action_space.sample() for i in range(switchboard.num_envs)]#[[switchboard.envs[i].envs[j].action_space.sample() for j in range(len(switchboard.envs[i].num_envs))] for i in range(switchboard.num_envs)]
    #     switchboard.step(a)

    # load pretrained model
    if load_model_path:
        model = ACER.load(load_model_path, env)

    # Create new model
    else:
        model = ACER(MlpLstmPolicy, env,
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

    return model, env


if __name__ == '__main__':
    model_save_path = 'experiments/actual/exp6/'

    # load train and test set
    scms = BoolSCMGenerator.load_dataset('data/scms/switchboard/3x3var_all.pkl')
    scms_train = scms[:20]  # exp6, training 2
    # scms_train = BoolSCMGenerator.make_obs_equ_3var_envs()  # exp 5 training
    model, board = train_switchboard_acer(4000000,
                                          train_scms=scms_train,
                                          workers=4,
                                          load_model_path='experiments/actual/exp6/model.zip',
                                          n_switches=3)

    model.save(model_save_path + 'model')

