from enum import Enum
import sys
sys.path.insert(0, '../common')
import helper_functions as hlp
from absl.flags import FLAGS
from pysc2.env import sc2_env

import test_GetUnits as t_units
import test_Distances as t_dist

def main():
    print()
    t_units.TestGetUnits()
    t_dist.TestDistances()
    

if __name__ == '__main__':
    FLAGS(sys.argv)
    main()