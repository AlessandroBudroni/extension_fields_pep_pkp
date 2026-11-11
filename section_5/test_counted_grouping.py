"""Independent checks for the counted construction used in Figure 2."""
import itertools
import math
import unittest

from counted_grouping import (counted_grouping_estimate,
                              optimize_counted_groupings,prepare_grouping_table)


class CountedGroupingTests(unittest.TestCase):
    def test_component_permanents(self):
        for g in range(1,8):
            for s in range(g):
                actual=sum(all(pi[i]!=0 for i in range(s))
                           for pi in itertools.permutations(range(g)))
                self.assertEqual(actual,(g-s)*math.factorial(g-1))

    def test_nu33_witness(self):
        result=counted_grouping_estimate(64,891,33)
        self.assertEqual(result.components,((16,3),)*4)
        self.assertEqual(result.selected_columns,1012)
        self.assertEqual(result.permutation_rank,124)
        self.assertEqual(result.residual_dimension,888)
        self.assertEqual(result.spare_equations,3)
        self.assertAlmostEqual(result.log2_trials,
                               math.log2(math.factorial(64))-4*math.log2(13*math.factorial(15)))

    def test_source_dimension_cap(self):
        prepare_grouping_table(8,36)
        result=counted_grouping_estimate(8,20,2)
        self.assertTrue(all(g<=2 for g,s in result.components))
        self.assertLessEqual(result.residual_dimension,17)

    def test_budget_monotonicity(self):
        optima=optimize_counted_groupings(9,64,max_group=7)
        self.assertTrue(all(a[0]<=b[0] for a,b in zip(optima,optima[1:])))
        for budget,(product,components,used) in enumerate(optima):
            self.assertEqual(sum(g for g,s in components),9)
            self.assertEqual(product,math.prod((g-s)*math.factorial(g-1) for g,s in components))
            self.assertEqual(used,sum((g-1)**2-s for g,s in components))
            self.assertLessEqual(used,budget)

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            counted_grouping_estimate(64,2,33,surplus=3)


if __name__=='__main__':
    unittest.main()
