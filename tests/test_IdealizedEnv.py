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
    build_drone = actions_by_name["BuildDrone"]
    select = actions_by_name["Select"]

    env = IdealizedSC2Env(game_loops_per_agent_step=10,
                          time_limit=100,
                          verbose=True)

    reward, state, game_end = env.reset()[0]
    while True:
        if select in state.available_actions:
            reward, state, game_end = env.step([lambda environ: select(environ, state.bases[0])])[0]
        if game_end:
            break
        if build_drone in state.available_actions:
            reward, state, game_end = env.step([lambda environ: build_drone(environ)])[0]
            print("Reward: {}".format(reward))
            print(state)
        if game_end:
            break


if __name__ == "__main__":
    test_idealized_env()
