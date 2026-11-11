"""Check native elimination, permutation elimination, and uniform references."""
import itertools
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from counted_rank_scaling import NativeRank,build_native,reduced_columns,uniform_reference
from counted_grouping_rank_experiment import binary_rank


@unittest.skipUnless(shutil.which("cc"), "native rank checks require a C compiler")
class RankScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory()
        cls.native=NativeRank(build_native(Path(cls.directory.name)))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_native_against_independent_python_rank(self):
        rng=random.Random(7)
        for bits in [0,1,7,63,64,65,128,137]:
            for count in [0,1,10,70,150]:
                columns=[rng.getrandbits(bits) for _ in range(count)]
                self.assertEqual(self.native(columns,bits),binary_rank(columns))
                self.assertEqual(self.native(columns+columns,bits),binary_rank(columns))

    def test_restricted_and_eliminated_nullities_match(self):
        rng=random.Random(33)
        for components in [((1,0),),((2,1),),((3,2),(2,1)),((4,1),(3,0))]:
            w=sum(g for g,s in components);m=9
            groups=[];cells=[];start=0
            for g,s in components:
                groups.append(list(range(start,start+g)))
                cells.append(list(range(start,start+g)));start+=g
            for _ in range(10):
                cols=[rng.getrandbits(m)|(1<<(m+i))|(1<<(m+w+j))
                      for i in range(w) for j in range(w)]
                omitted=[rng.choice(cell) for cell in cells]
                chosen=[i*w+j for group,cell,(g,s),z in zip(groups,cells,components,omitted)
                        for local,i in enumerate(group) for j in cell if local>=s or j!=z]
                full_nullity=len(chosen)-binary_rank(cols[i] for i in chosen)
                reduced=reduced_columns(cols,groups,cells,components,omitted,w,m)
                self.assertEqual(len(reduced),sum((g-1)**2-s for g,s in components))
                self.assertEqual(len(reduced)-self.native(reduced,m),full_nullity)

    def test_uniform_references_by_exhaustive_matrices(self):
        for m,c in [(0,0),(2,1),(2,2),(3,2),(2,3)]:
            nullities=[c-binary_rank(cols) for cols in itertools.product(range(1<<m),repeat=c)]
            mean,prob=uniform_reference(m,c)
            self.assertAlmostEqual(mean,sum(2**d for d in nullities)/len(nullities))
            self.assertAlmostEqual(prob,sum(d==0 for d in nullities)/len(nullities))


if __name__=='__main__':
    unittest.main()
