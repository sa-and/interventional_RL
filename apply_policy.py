from stable_baselines import ACER
from Environments import BoolSCMGenerator, Switchboard
from Agents import DiscreteSwitchboardAgent
from episode_evals import NoEval


# def create_switchboard_acer_fixed():
#     gen = BoolSCMGenerator(5, 0)
#     agent = DiscreteSwitchboardAgent(3)
#     eval_func = NoEval()
#     a = Switchboard(agent, scm=BoolSCMGenerator.make_obs_equ_3var_envs()[1], eval_func=eval_func)
#     return a


if __name__ == '__main__':
    model_path = f'experiments/actual/exp5/model.zip'
    model = ACER.load(model_path)
    model_workers = model.n_envs
    n_vars = 3

    # just do this multiple times for easier inspection
    for j in range(20):
        test_evn = a = Switchboard(agent=DiscreteSwitchboardAgent(n_vars),
                                   scm=BoolSCMGenerator.make_obs_equ_3var_envs()[1],
                                   eval_func=NoEval())
        states = model.initial_state
        done = [False for i in range(model_workers)]
        obs = test_evn.reset()
        obs = [obs for i in range(model_workers)]

        for i in range(49):
            print(obs)
            actions, states = model.predict(obs, state=states, mask=done, deterministic=True)
            print(test_evn.agent.get_action_from_actionspace_sample(actions[0]))
            obs, _, done, _ = test_evn.step(actions[0])
            obs = [obs for i in range(model_workers)]
            done = [done for i in range(model_workers)]
            test_evn.render()
        test_evn.agent.display_causal_model()
        print('\n\n\n\n')
