from collections import deque


class ActionSpace:
    choices = [
        (lambda i: lambda q: q.append(i))(i) for i in range(3)
    ]
    action_spec = {
        'num_actions': len(choices)
    }

    def __init__(self):
        self.actionq = deque([])

    def act(self, choice, _=None, __=None):
        if choice in range(len(ActionSpace.choices)):
            return 1, ActionSpace.choices[choice](self.actionq)
        else:
            return -1, ActionSpace.choices[0](self.actionq)

    def action_step(self):
        if self.actionq:
            action = self.actionq.popleft()
        else:
            action = 0
        return action
