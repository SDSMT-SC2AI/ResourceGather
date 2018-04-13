from sc2_ez_model.environment.model import IdealizedSC2Env

actions_by_name = {a.__name__: a for a in IdealizedSC2Env.actions}


def test_idealized_env():
    basic_test()
    build_drones()


def basic_test():
    env = IdealizedSC2Env(game_loops_per_agent_step=1000,
                          time_limit=100,
                          verbose=True)
    print("Should collect 1500 minerals")
    env.step([lambda environ: actions_by_name['NoOp'](environ)])
    print(env)


def build_drones():
    locals().update(actions_by_name)
    # print(locals())
    print(state.available_actions)
    exit()
    env = IdealizedSC2Env(game_loops_per_agent_step=10,
                          time_limit=100,
                          verbose=True)

    reward, state, game_end = env.reset()[0]
    if Select in state.available_actions:
            reward, state, game_end = env.step([lambda environ: Select(environ, state.bases[0])])[0]
    while not game_end:
        if BuildDrone in state.available_actions:
            reward, state, game_end = env.step([lambda environ: BuildDrone(environ)])[0]
            print("Reward: {}".format(reward))
            print(state)
        else:
            reward, state, game_end = env.step([lambda environ: BuildOverlord(environ)])[0]


if __name__ == "__main__":
    test_idealized_env()
