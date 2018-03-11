from enum import Enum
import sys
sys.path.insert(0, '../common')
import helper_functions as hlp
from pysc2.env import sc2_env
from absl.flags import FLAGS

def TestDistances():
    environ = sc2_env.SC2Env(map_name="test_Distances", agent_race="Z")
    _obs = environ._parallel.run(c.observe for c in environ._controllers)

    marine = hlp.GetUnits([48],_obs, hlp.Alliance.Self)     # 0,0
    marauder = hlp.GetUnits([51],_obs, hlp.Alliance.Self)   # 4,3
    ling = hlp.GetUnits([105], _obs, hlp.Alliance.Self)     # -3,0

    print("Distance tests:")
    print("-----------------------------------------------------")    
    if ( hlp.DistSquared(marine[0].pos, marauder[0].pos) != 25.0):
        print("\nFailed to measure DistSquared\n")
        exit()
    else:
        print("DistSquared passed")

    if ( hlp.InDistSqRange(marine[0].pos, ling[0].pos, 8.9) ):
        print("\nFailed InDistSqRange\n")
        exit()
    else:
        print("InDistSqRange passed")

    if ( hlp.InRadius(marine[0].pos, ling[0].pos, 2.0) ):
        print("\nFailed InRadius\n")
        exit()
    else:
        print("InRadius passed")

    print("Distances passed all tests\n") 

def main():
    TestDistances()
    

if __name__ == '__main__':
    FLAGS(sys.argv)
    main()


