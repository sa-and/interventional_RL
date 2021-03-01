from environments.Switchboard import Switchboard

swtchbrd = Switchboard()
for i in range(10000):
    rnd_action = swtchbrd.action_space.sample()
    swtchbrd.step(rnd_action)
    swtchbrd.render()
print(swtchbrd.agent.get_est_postint_distrib('x0', (2, True)))
print()
print(swtchbrd.agent.get_est_postint_distrib('x0', (2, False)))
print()
print(swtchbrd.agent.get_est_avg_causal_effect('x0', (2, True), (2, False)))
print()
print(swtchbrd.agent.get_est_cond_distr('x0', ('x2', True)))
print()
print(swtchbrd.agent.get_est_cond_distr('x0', ('x2', False)))
