"""Independent small exact checks of the original KMP estimator."""
import math
import unittest
from fractions import Fraction
from kmp import kmp_complexity, components_log2
from plot_pkp_attacks_vs_n import largest_k_below_one


def exact_terms(n,r,ell,q,u,v):
    a=math.factorial(n)//math.factorial(n-u)
    b=math.factorial(n)//math.factorial(n-v)
    factor=Fraction(q)**(ell*(n-r-u-v))
    return [Fraction(a),Fraction(b),a*b*factor,
            Fraction(math.factorial(n)//math.factorial(n-u-v))*factor]


class KmpEstimatorTests(unittest.TestCase):
    def test_terms_against_exact_arithmetic(self):
        actual=components_log2(7,1,3,2,3,4)
        self.assertAlmostEqual(actual['T_log2'],math.log2(sum(exact_terms(7,3,1,4,2,3))))

    def test_optimization_against_exhaustive_exact_cost(self):
        for n,r,ell,q in [(5,2,1,4),(7,3,1,8),(6,3,2,4)]:
            candidates=[sum(exact_terms(n,r,ell,q,u,v)) for u in range(n+1)
                        for v in range(n-u+1) if u+v>=n-r+1]
            self.assertAlmostEqual(kmp_complexity(n,r,ell,q)[1],math.log2(min(candidates)))

    def test_paper_parameters(self):
        self.assertAlmostEqual(kmp_complexity(64,27,1,2**33)[1],120.6924037809325)
        self.assertAlmostEqual(kmp_complexity(64,27,1,2**11)[1],143.51298200194614)

    def test_additional_solution_parameter_selection(self):
        self.assertEqual(largest_k_below_one(2,1,1),1)
        for n in [20,64,160]:
            k=largest_k_below_one(n,11,1)
            numerator=math.factorial(n)-1
            self.assertLess(numerator,2**(11*(n-k)))
            self.assertGreaterEqual(numerator,2**(11*(n-k-1)))


if __name__=='__main__':
    unittest.main()
