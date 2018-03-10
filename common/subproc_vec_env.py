## based on chris-chris/pysc2-examples @ github.com
import numpy as np
from argparse import Namespace
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv
from pysc2.env import environment
from pysc2.env import sc2_env

def worker(remote, agent, map_name, flags, i):
    def step(d):
        parent = Namespace(d)
        parent.reward = 0
        result = env.step(actions=parent.data)
        parent.reward += parent.results[0]
        parent.done = results[0].step_type == environment.StepType.LAST



    with sc2_env.SC2Env(
            map_name=map_name,
            agent_race=flags.agent_race,
            bot_race=flags.bot_race,
            difficulty=flags.difficulty,
            step_mul=flags.step_mul,
            game_steps_per_episode=flags.game_steps_per_episode,
            screen_size_px=(flags.screen_resolution, flags.screen_resolution),
            minimap_size_px=(flags.minimap_resolution, flags.minimap_resolution),
            visualize=False
    ) as env:
        total_frams = 0

        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                reward = 0

                if len(group_list) == 0 or common.check_group_list(env, result):
                    print("init group list")
                    result, xy_per_marine = common.init(env, result)
                    group_list = common.update_group_list(result)

                action1 = data[0][0]
                action2 = data[0][1]
                # func = actions.FUNCTIONS[action1[0]]
                # print("agent(",i," ) action : ", action1, " func : ", func)
                func = actions.FUNCTIONS[action2[0]]
                # print("agent(",i," ) action : ", action2, " func : ", func)

                result = env.step(actions=[action1])
                reward += result[0].reward
                done = result[0].step_type == environment.StepType.LAST

                move = True

                if len(action2[1]) == 2:
                    x, y = action2[1][1]
                    # print("x, y:", x, y)

                    # if x == 0 and y == 0:
                    #   move = False

                if (331 in available_actions and move and not done):
                    try:
                        result = env.step(actions=[action2])
                        reward += result[0].reward
                        done = result[0].step_type == environment.StepType.LAST
                    except Exception as e:
                        print("e :", e)

                ob = (result[0].observation["screen"][
                      _PLAYER_RELATIVE:_PLAYER_RELATIVE + 1] == 3).astype(int)

                #  (1, 32, 32)
                selected = result[0].observation["screen"][
                           _SELECTED:_SELECTED + 1]  # (1, 32, 32)
                # extra = np.zeros((1, 32, 32))
                control_groups = result[0].observation["control_groups"]
                army_count = env._obs[0].observation.player_common.army_count

                available_actions = result[0].observation["available_actions"]
                info = result[0].observation["available_actions"]
                if done:
                    result = env.reset()

                    if len(group_list) == 0 or common.check_group_list(env, result):
                        # print("init group list")
                        result, xy_per_marine = common.init(env, result)
                        group_list = common.update_group_list(result)

                    info = result[0].observation["available_actions"]

                if len(action1[1]) == 2:

                    group_id = action1[1][1][0]

                    player_y, player_x = (result[0].observation["screen"][
                                              _SELECTED] == 1).nonzero()

                    if len(player_x) > 0:
                        if (group_id == 1):
                            xy_per_marine["1"] = [int(player_x.mean()), int(player_y.mean())]
                        else:
                            xy_per_marine["0"] = [int(player_x.mean()), int(player_y.mean())]

                remote.send((ob, reward, done, info, army_count,
                             control_groups, selected, xy_per_marine))

            elif cmd == 'reset':
                result = env.reset()
                reward = 0

                if len(group_list) == 0 or common.check_group_list(env, result):
                    # print("init group list")
                    result, xy_per_marine = common.init(env, result)
                    group_list = common.update_group_list(result)

                reward += result[0].reward
                ob = (result[0].observation["screen"][
                      _PLAYER_RELATIVE:_PLAYER_RELATIVE + 1] == 3).astype(int)
                selected = result[0].observation["screen"][
                           _SELECTED:_SELECTED + 1]  # (1, 32, 32)
                # extra = np.zeros((1, 32, 32))
                control_groups = result[0].observation["control_groups"]
                army_count = env._obs[0].observation.player_common.army_count

                done = result[0].step_type == environment.StepType.LAST
                info = result[0].observation["available_actions"]
                available_actions = result[0].observation["available_actions"]
                remote.send((ob, reward, done, info, army_count,
                             control_groups, selected, xy_per_marine))
            elif cmd == 'close':
                print("worker", i, "finished")
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.action_spec().functions[data], ""))
            elif cmd == "action_spec":
                remote.send((env.action_spec().functions[data]))
            else:
                raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, num_envs, map_name, flags):
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])

        self.ps = []
        i = 0
        for (work_remote,) in zip(self.work_remotes, ):
            self.ps.append(
                Process(target=worker, args=(work_remote, map_name, flags, i)))
            i += 1

        #
        # self.ps = [Process(target=worker, args=(work_remote, (map_name)))
        #            for (work_remote,) in zip(self.work_remotes,)]
        for p in self.ps:
            p.start()

        self.remotes[0].send(('get_spaces', 1))
        self.action_space, self.observation_space = self.remotes[0].recv()
        # print("action_space: ", self.action_space, " observation_space: ", self.observation_space)

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', [action]))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos, army_counts, control_groups, selected, xy_per_marine = zip(
            *results)
        return np.stack(obs), np.stack(rews), np.stack(
            dones), infos, army_counts, control_groups, np.stack(
            selected), xy_per_marine

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos, army_counts, control_groups, selected, xy_per_marine = zip(
            *results)
        return np.stack(obs), np.stack(rews), np.stack(
            dones), infos, army_counts, control_groups, np.stack(
            selected), xy_per_marine

    def action_spec(self, base_actions):
        for remote, base_action in zip(self.remotes, base_actions):
            remote.send(('action_spec', base_action))
        results = [remote.recv() for remote in self.remotes]

        return results

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)



def GetUnits(unit_type, obs):
    """Gets all units of unit_type(s), unit_type MUST be a list"""
    ret_val = []
    if (isinstance(unit_type, int)):
        for item in obs[0].observation.raw_data.units:
            if (item.unit_type == unit_type):
                ret_val.append(item)
    else:
        for item in obs[0].observation.raw_data.units:
            if (item.unit_type in unit_type):
                ret_val.append(item)
    return ret_val