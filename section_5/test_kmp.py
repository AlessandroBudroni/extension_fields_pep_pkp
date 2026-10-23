"""Small exact combinatorial checks for the SPECK estimator (stdlib only)."""
import itertools
import math
import unittest

from kmp import arrangement_counts, speck_complexity, speck_random_profile_complexity


class SpeckEstimatorTests(unittest.TestCase):
    def test_arrangements_against_exhaustive_enumeration(self):
        for profile in [(2, 2, 1), (3, 1), (1, 1, 1, 1)]:
            vector = [i for i, m in enumerate(profile) for _ in range(m)]
            expected = [len(set(itertools.permutations(vector, t)))
                        for t in range(len(vector) + 1)]
            self.assertEqual(arrangement_counts(profile), tuple(expected))

    def test_speck_formula_and_optimization(self):
        # Independently evaluate Prop. 2 / Lemma 2 using ordinary arithmetic.
        profile = (2, 2, 1)
        n, k, q = 5, 2, 4
        counts = arrangement_counts(profile)
        solutions = 1 + (counts[n] - 1) / q ** (n - k)
        candidates = []
        for t in range(1, n - k + 1):
            u1 = (k + t) // 2
            u2 = k + t - u1
            a = max(1, counts[u1] / math.sqrt(solutions))
            b = max(1, counts[u2] / math.sqrt(solutions))
            candidates.append((math.log2(max(a, b, a*b/q**t)), t,
                               math.log2(a+b+a*b/q**t)))
        expected = min(candidates)
        actual = speck_complexity(n, k, q, profile)
        self.assertAlmostEqual(actual['best_log2_T'], expected[0])
        self.assertEqual(actual['matching_symbols'], expected[1])
        self.assertAlmostEqual(actual['total_log2_T'], expected[2])
        self.assertAlmostEqual(actual['random_guess_log2'], math.log2(counts[n]/solutions))

    def test_unique_solution_limit(self):
        result = speck_complexity(6, 3, 2**100, (1,)*6)
        self.assertLess(result['log2_solutions'], 1e-12)
        self.assertAlmostEqual(result['best_log2_T'], math.log2(6*5))

    def test_reproducibility_and_large_fields(self):
        a = speck_random_profile_complexity(12, 6, 4, samples=8, seed=2)
        self.assertEqual(a, speck_random_profile_complexity(12, 6, 4, samples=8, seed=2))
        self.assertLessEqual(a['min_log2_T'], a['best_log2_T'])
        self.assertLessEqual(a['best_log2_T'], a['max_log2_T'])
        self.assertTrue(math.isfinite(speck_complexity(64, 37, 2**60, (1,)*64)['best_log2_T']))

    def test_reject_invalid_profiles(self):
        for profile in [(2, 2), (0, 5), (1,)*5]:
            with self.assertRaises(ValueError):
                speck_complexity(5, 2, 4, profile)


if __name__ == '__main__':
    unittest.main()
