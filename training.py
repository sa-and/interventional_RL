from typing import List
from environments import SCMEnvironment, SCMEnvironmentReservoir
from agents import DiscreteAgent
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

    board = SCMEnvironmentReservoir(train_scms,
                                    n_switches,
                                    DiscreteAgent,
                                    FixedLengthEpisode)

    # data collection phase
    board.collect_interv_data(80)
    print('data collection phase done\n\n\n\n\n\n\n\n\n\n')

    env = venv.SubprocVecEnv([lambda: copy.deepcopy(board) for w in range(workers)], start_method='spawn')

    # load pretrained model
    if load_model_path:
        model = ACER.load(load_model_path, env)

    # Create new model
    else:
        model = ACER(MlpLstmPolicy, env,
                     policy_kwargs={'net_arch': [50,
                                                 'lstm',
                                                 {'pi': [40],
                                                  'vf': [10]}],
                                    'n_lstm': 120},

                     n_steps=30,
                     n_cpu_tf_sess=8,
                     replay_ratio=10,
                     buffer_size=500000,
                     gamma=0.999
                     )

    model.learn(steps)

    return model, env


if __name__ == '__main__':
    model_save_path = 'experiments/actual/exp9/'  # fixed is on training 2

    # load train and test set
    # exp8, training
    scms = BoolSCMGenerator.load_dataset('data/scms/Dasgupta/4x1_25000.pkl')
    scms_train = scms[:10000]
    model, board = train_switchboard_acer(10000000,
                                          train_scms=scms_train,
                                          workers=4,
                                          load_model_path='experiments/actual/exp9/model.zip',
                                          n_switches=4)

    model.save(model_save_path + 'model')

