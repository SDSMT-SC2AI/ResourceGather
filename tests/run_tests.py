import context
import sys
sys.path.insert(0, '../common')
from absl.flags import FLAGS

import test_Policy as t_policy
import test_GetUnits as t_units
import test_Distances as t_dist

def main():
    print()
    t_policy.test_policy()
    t_units.TestGetUnits()
    t_dist.TestDistances()
    

if __name__ == '__main__':
    FLAGS(sys.argv)
    main()