from stable_baselines import ACER
from environments import SCMEnvironment
from scm import BoolSCMGenerator, CausalGraphGenerator, SCMGenerator, DasguptaSCMGenerator
from agents import DiscreteAgent
from episode_evals import NoEval
import networkx as nx
import numpy as np
import cdt
from cdt.causality.graph import PC, GES, GIES

from pandas import DataFrame
cdt.SETTINGS.rpath = 'C:/Program Files/R/R-4.0.5/bin/Rscript'


def apply_policy(model, test_scm, n_vars, episode_length, display, env_type='Switchboard'):
    model_workers = model.n_envs
    test_evn = SCMEnvironment(agent=DiscreteAgent(n_vars, env_type=env_type),
                              scm=test_scm,
                              eval_func=NoEval())
    # just do this multiple times for easier inspection
    states = model.initial_state
    done = [False for i in range(model_workers)]
    obs = test_evn.reset()
    obs = [obs for i in range(model_workers)]

    for i in range(episode_length-1):
        print(obs)
        actions, states = model.predict(obs, state=states, mask=done, deterministic=True)
        print(test_evn.agent.get_action_from_actionspace_sample(actions[0]))
        obs, _, done, _ = test_evn.step(actions[0])
        obs = [obs for i in range(model_workers)]
        done = [done for i in range(model_workers)]
        test_evn.render()
    if display:
        test_evn.agent.display_causal_model()
        print('\n\n\n\n')
    return nx.DiGraph(test_evn.agent.causal_model)


def compare_graphs(predicted: nx.DiGraph, target: nx.DiGraph) -> int:
    assert len(predicted.nodes) == len(target.nodes), 'Graphs need to have the same amount of nodes'
    
    differences = 0
    for node in predicted.adj:
        # check which edges are too much in the predicted graph
        for parent in predicted.adj[node]:
            if not parent.upper() in target.adj[node.upper()]:
                differences += 1
        # check which edges are missing in the predicted graph
        for parent in target.adj[node.upper()]:
            if not parent.lower() in predicted.adj[node]:
                differences += 1
                
    return differences


def learn_from_obs(algo, scm):
    columns = ['X0', 'X1', 'X2']
    obs_data = DataFrame(columns=columns)
    for i in range(500):
        vars = scm.get_next_instantiation()[0]
        obs_data = obs_data.append({'X' + str(i): float(vars[i]) for i in range(len(vars))},
                                   ignore_index=True)
    algo = algo()
    return algo.predict(obs_data)


if __name__ == '__main__':
    model = ACER.load(f'experiments/actual/exp9/model.zip')
    eval_data = SCMGenerator.load_dataset('data/scms/Dasgupta/4x1_25000.pkl')[:50]
    differences = []
    for scm in eval_data:
        target_graph = CausalGraphGenerator.create_graph_from_scm(scm)

        for run in range(10):
            # from learned policy
            predicted_graph = apply_policy(model=model,
                                           test_scm=scm,
                                           n_vars=4,
                                           episode_length=30,
                                           display=False,
                                           env_type='Dasgupta')

            # from obs based algo
            # predicted_graph = learn_from_obs(algo=GIES, scm=scm)

            # random
            # predicted_graph = CausalGraphGenerator.create_graph_from_scm(gen.create_random()[0])

            difference = compare_graphs(predicted_graph, target_graph)
            differences.append(difference)
            print('.')

    differences = np.array(differences)
    print('mean:', differences.mean())
    print('std:', differences.std())
