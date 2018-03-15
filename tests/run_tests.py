import context
import sys
from absl.flags import FLAGS

import tests.test_Policy as t_policy
import tests.test_GetUnits as t_units
import tests.test_Distances as t_dist

def main():
    print()
    t_policy.test_policy()
    t_units.TestGetUnits()
    t_dist.TestDistances()
    

if __name__ == '__main__':
    FLAGS(sys.argv)
    main()