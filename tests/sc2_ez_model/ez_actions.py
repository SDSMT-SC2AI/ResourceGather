choices = [[0], [1], [2]]
action_spec = {
    'number of actions': len(choices)
}


class Action_Space:
    def __init__(self):
        self.actionq = deque(["No_Op"]*10)
        self.pointq = deque([])

    def act(choice):
        if choice in range(len(choices)):
            return 1, choices[choice]
        else:
            return -1, choices[0]
