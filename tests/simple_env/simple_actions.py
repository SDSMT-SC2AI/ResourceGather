choices = [[0], [1], [2]]
action_spec = {
    'num_actions': len(choices)
}


def act(choice, env_obs = None):
    if choice in range(len(choices)):
        return 1, choices[choice]
    else:
        return -1, choices[0]
