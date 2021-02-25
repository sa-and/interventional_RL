class SwitchboardAgent:
    def __init__(self, n_switches):
        # create a dictonary of actions that can be performed on the switchboard
        self.actions = [(i, True) for i in range(n_switches)]
        self.actions.extend([(i, False) for i in range(n_switches)])
        self.actions.append((None, None))
