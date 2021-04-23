from typing import List
from Environments import Switchboard, SwitchboardReservoir
from Agents import DiscreteSwitchboardAgent
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines import ACER
import stable_baselines.common.vec_env as venv
from scm import StructuralCausalModel, BoolSCMGenerator
import copy
from episode_evals import FixedLengthEpisode, TwoPhaseFixedEpisode


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

    # load pretrained model
    if load_model_path:
        model = ACER.load(load_model_path, env)

    # Create new model
    else:
        model = ACER(MlpLstmPolicy, env,
                     policy_kwargs={'net_arch': [40,
                                                 'lstm',
                                                 {'pi': [40],
                                                  'vf': [10]}],
                                    'n_lstm': 50},

                     n_steps=5,
                     n_cpu_tf_sess=8,
                     replay_ratio=10,
                     buffer_size=500000,
                     gamma=0.999
                     )

    model.learn(steps)

    return model, env


if __name__ == '__main__':
    model_save_path = 'experiments/actual/exptest/'

    # load train and test set
    # exp8, training
    scms = BoolSCMGenerator.load_dataset('data/scms/switchboard/4x4var_all.pkl')
    scms_train = scms[:50]
    scms_train = list(BoolSCMGenerator.make_obs_equ_3var_envs())
    model, board = train_switchboard_acer(1000000,
                                          train_scms=scms_train,
                                          workers=8,
                                          load_model_path=None,
                                          n_switches=3)

    model.save(model_save_path + 'model')

