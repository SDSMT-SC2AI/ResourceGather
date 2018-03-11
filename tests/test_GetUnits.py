from enum import Enum
import sys
sys.path.insert(0, '../common')
import helper_functions as hlp
from pysc2.env import sc2_env
from absl.flags import FLAGS

def TestGetUnits():
    environ = sc2_env.SC2Env(map_name="test_GetUnits", agent_race="Z")
    _obs = environ._parallel.run(c.observe for c in environ._controllers)

    # Single type of friendly unit
    marines = hlp.GetUnits([48],_obs, hlp.Alliance.Self)

    # Single type of enemy unit
    lings = hlp.GetUnits([105], _obs, hlp.Alliance.Enemy)

    # Multi-type of friendly units
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

def main():
    TestGetUnits()

if __name__ == '__main__':
    FLAGS(sys.argv)
    main()


