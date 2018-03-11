from enum import Enum
import sys
sys.path.insert(0, '../common')
import helper_functions as hlp
from absl import flags
from absl.flags import FLAGS
from pysc2.env import sc2_env

def main():
    map_name = FLAGS.map_name
    environ = sc2_env.SC2Env(map_name=map_name, agent_race="Z")
    _obs = environ._parallel.run(c.observe for c in environ._controllers)



    marines = hlp.GetUnits([48],_obs, hlp.Alliance.Self)
    lings = hlp.GetUnits([105], _obs, hlp.Alliance.Enemy)
    player1 = hlp.GetUnits([48, 51],_obs, hlp.Alliance.Self)


    if (len(marines) != 3):
        print("\nFailed to get self-units\n")
        exit()

    if (len(lings) != 2):
        print("\nFailed to get enemy-units\n")
        exit()

    if (len(player1) != 5):
        print("\nFailed to get multiple-units\n")
        exit()   

    print("GetUnits() passed all tests")

if __name__ == '__main__':
    flags.DEFINE_string("map_name", "test_GetUnits", "Name of the map/Unit Test")
    FLAGS(sys.argv)
    main()


