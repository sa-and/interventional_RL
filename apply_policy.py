from stable_baselines import DQN, ACER
from stable_baselines.a2c import A2C
from Environments import BoolSCMGenerator, Switchboard
from Agents import DiscreteSwitchboardAgent
import stable_baselines.common.vec_env as venv
import numpy as np
from training import load_dataset


def create_switchboard_acer_fixed():
    gen = BoolSCMGenerator(5, 0)
    agent = DiscreteSwitchboardAgent(5)
    a = Switchboard(agent, fixed_episode_length=True, scm=gen.make_switchboard_scm_with_context())
    return a


def load_policy(path, env, algo='ACER'):
    if algo == 'ACER':
        return ACER.load(path, env)
    elif algo == 'DQN':
        return DQN.load(path, env)
    elif algo == 'A2C':
        return A2C.load(path, env)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    model_path = f'experiments/actual/exp2/model.zip'
    model = ACER.load(model_path)
    model_workers = model.n_envs
    scms = load_dataset('data/scms/switchboard/5x0var_25000.pkl')

    # just do this multiple times for easier inspection
    for j in range(20):
        test_evn = create_switchboard_acer_fixed()
        states = None
        done = [False for i in range(model_workers)]
        obs = [test_evn.observation for i in range(model_workers)]

        for i in range(50):
            print(obs)
            actions, states = model.predict(obs, state=states, mask=done, deterministic=True)
            print(test_evn.agent.get_action_from_actionspace_sample(actions[0]))
            obs, _, done, _ = test_evn.step(actions[0])
            obs = [obs for i in range(model_workers)]
            done = [done for i in range(model_workers)]
            test_evn.render()
        test_evn.agent.display_causal_model()
    print(done)