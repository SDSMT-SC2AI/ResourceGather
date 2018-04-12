choices = [[0], [1], [2]]
action_spec = {
    'number of actions': len(choices)
}


def act(choice):
    if choice in range(len(choices)):
        return 1, choices[choice]
    else:
        return -1, choices[0]
