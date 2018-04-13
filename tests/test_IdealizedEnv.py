from sc2_ez_model.environment.model import IdealizedSC2Env

globals().update({a.__name__: a for a in IdealizedSC2Env.actions})
# actions_by_name = {a.__name__: a for a in IdealizedSC2Env.actions}


def test_idealized_env():
    basic_test()
    build_drones()


def basic_test():
    env = IdealizedSC2Env(game_loops_per_agent_step=1000,
                          time_limit=100,
                          verbose=False)
    print("Should collect 1500 minerals")
    env.step([lambda environ: NoOp(environ)])
    print(env)


def build_drones():
    env = IdealizedSC2Env(game_loops_per_agent_step=10,
                          time_limit=720,
                          verbose=True)

    reward, state, game_end = env.reset()[0]
    max_bases = 6
    bases = 0
    real_drone_count = 0
    pool_constructing = False

    while not game_end:
        acted = False
        # Select a base
        if Select in state.available_actions:
            reward, state, game_end = env.step([lambda environ: Select(environ, state.bases[0])])[0]
            acted = True
        if BuildDrone in state.available_actions:
            reward, state, game_end = env.step([lambda environ: BuildDrone(environ)])[0]
            acted = True
            real_drone_count += 1
        elif BuildOverlord in state.available_actions:
            reward, state, game_end = env.step([lambda environ: BuildOverlord(environ)])[0]
            acted = True
        elif BuildQueen in state.available_actions:
            reward, state, game_end = env.step([lambda environ: BuildQueen(environ)])[0]
            acted = True

        if InjectLarva in state.available_actions:
            reward, state, game_end = env.step([lambda environ: InjectLarva(environ)])[0]
            acted = True
                
        # Select a mineral patch
        if Select in state.available_actions:
            reward, state, game_end = env.step([lambda environ: Select(environ, state.bases[0].minerals)])[0]
        if bases != max_bases and BuildBase in state.available_actions:
            reward, state, game_end = env.step([lambda environ: BuildBase(environ)])[0]
            # print("Reward: {}".format(reward))
            print(state)  
            bases += 1
            acted = True
        elif not pool_constructing and state.time_elapsed > 60 and not state.spawning_pool and BuildSpawningPool in state.available_actions:
            reward, state, game_end = env.step([lambda environ: BuildSpawningPool(environ)])[0]
            pool_constructing = True



        if not acted:
            reward, state, game_end = env.step([lambda environ: NoOp(environ)])[0]
    print(state)
    print("real drones: ", real_drone_count)



if __name__ == "__main__":
    test_idealized_env()
