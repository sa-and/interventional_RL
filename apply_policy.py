from stable_baselines import DQN, ACER
from stable_baselines.a2c import A2C
from Environments import BoolSCMGenerator, Switchboard
from Agents import DiscreteSwitchboardAgent
import stable_baselines.common.vec_env as venv
import numpy as np


def create_switchboard_acer_fixed():
    gen = BoolSCMGenerator(5, 0)
    agent = DiscreteSwitchboardAgent(5)
    a = Switchboard(agent, fixed_episode_length=True, scm=gen.create_random()[0])
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


model_path = f'experiments/preliminary/exp19/model.zip'
switchboard = venv.DummyVecEnv([create_switchboard_acer_fixed for i in range(8)])
obs = switchboard.reset()
model = load_policy(model_path, switchboard)
state = None
done = [False for _ in range(8)]
for i in range(50):
    print(switchboard.envs[0].observation)
    action, state = model.predict(obs, state=state, deterministic=True, mask=done)
    # print(agent.get_action_from_actionspace_sample(action[0]))
    obs, _, done, _ = switchboard.step(action)
    switchboard.envs[0].render()
    print()
print(done)
switchboard.envs[0].agent.display_causal_model()