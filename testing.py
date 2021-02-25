from environments.Switchboard import Switchboard

swtchbrd = Switchboard()
for i in range(500):
    rnd_action = swtchbrd.action_space.sample()
    swtchbrd.step(rnd_action)
    swtchbrd.render()
print(swtchbrd.agent.get_est_postint_distrib('x3', (4, True)))
print(swtchbrd.agent.get_est_postint_distrib('x3', (4, False)))
print(swtchbrd.agent.get_est_postint_distrib('x3', (None, None)))
print(swtchbrd.agent.get_est_avg_causal_effect('x3', (4, True), (4, False)))
