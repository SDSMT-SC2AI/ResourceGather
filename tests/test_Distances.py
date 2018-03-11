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



    marine = hlp.GetUnits([48],_obs, hlp.Alliance.Self)     # 0,0
    marauder = hlp.GetUnits([51],_obs, hlp.Alliance.Self)   # 4,3
    ling = hlp.GetUnits([105], _obs, hlp.Alliance.Self)     # -3,0

    
    if ( hlp.DistSquared(marine[0].pos, marauder[0].pos) != 25.0):
        print("\nFailed to measure DistSquared\n")
        exit()

    if ( hlp.InDistSqRange(marine[0].pos, ling[0].pos, 8.9) ):
        print("\nFailed InDistSqRange\n")
        exit()

    if ( hlp.InRadius(marine[0].pos, ling[0].pos, 2.0) ):
        print("\nFailed InRadius\n")
        exit()

    print("Distances() passed all tests")
    

if __name__ == '__main__':
    flags.DEFINE_string("map_name", "test_Distances", "Name of the map/Unit Test")
    FLAGS(sys.argv)
    main()


