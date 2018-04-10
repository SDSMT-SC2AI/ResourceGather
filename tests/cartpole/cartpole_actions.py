from collections import deque


class ActionSpace:
    choices = [
        (lambda i: lambda q: q.append(i))(i) for i in range(2)
    ]
    action_spec = {
        'num_actions': len(choices)
    }

    def __init__(self):
        self.actionq = deque([])

    def act(self, choice, _=None, __=None):
        if choice in range(len(ActionSpace.choices)):
            ActionSpace.choices[choice](self.actionq)
            return 1, None
        else:
            ActionSpace.choices[0](self.actionq)
            return -1, None

