# This source code is licensed under the license found in the
# LICENSE file in the {root}/models/Private_PGM/ directory of this source tree.
import unittest
from models.Private_PGM.mbi.factor import Factor
from models.Private_PGM.mbi.domain import Domain
import numpy as np

class TestFactor(unittest.TestCase):

    def setUp(self):
        attrs = ['a','b','c']
        shape = [2,3,4]
        domain = Domain(attrs, shape)
        values = np.random.rand(*shape)
        self.factor = Factor(domain, values)
   
    def test_expand(self):
        domain = Domain(['a','b','c','d'], [2,3,4,5])
        res = self.factor.expand(domain)
        self.assertEqual(res.domain, domain)
        self.assertEqual(res.values.shape, domain.shape)

        res = res.sum(['d']) * 0.2
        self.assertTrue(np.allclose(res.values, self.factor.values))

    def test_transpose(self):
        attrs = ['b','c','a']
        tr = self.factor.transpose(attrs)
        ans = Domain(attrs, [3,4,2])
        self.assertEqual(tr.domain, ans)

    def test_project(self):
        res = self.factor.project(['c','a'], agg='sum')
        ans = Domain(['c','a'], [4,2])
        self.assertEqual(res.domain, ans)
        self.assertEqual(res.values.shape, (4,2))

        res = self.factor.project(['c','a'], agg='logsumexp')
        self.assertEqual(res.domain, ans)
        self.assertEqual(res.values.shape, (4,2))

    def test_sum(self):
        res = self.factor.sum(['a','b'])
        self.assertEqual(res.domain, Domain(['c'],[4]))
        self.assertTrue(np.allclose(res.values, self.factor.values.sum(axis=(0,1))))

    def test_logsumexp(self):
        res = self.factor.logsumexp(['a','c'])
        values = self.factor.values
        ans = np.log(np.sum(np.exp(values), axis=(0,2)))
        self.assertEqual(res.domain, Domain(['b'],[3]))
        self.assertTrue(np.allclose(res.values, ans))

    def test_binary(self):
        dom = Domain(['b','d','e'], [3,5,6])
        vals = np.random.rand(3,5,6)
        factor = Factor(dom, vals)
        
        res = self.factor * factor
        ans = Domain(['a','b','c','d','e'], [2,3,4,5,6])
        self.assertEqual(res.domain, ans)

        res = self.factor + factor
        self.assertEqual(res.domain, ans)

        res = self.factor * 2.0
        self.assertEqual(res.domain, self.factor.domain)
        
        res = self.factor + 2.0
        self.assertEqual(res.domain, self.factor.domain)
       
        res = self.factor - 2.0
        self.assertEqual(res.domain, self.factor.domain)
 
        res = self.factor.exp().log()
        self.assertEqual(res.domain, self.factor.domain)
        self.assertTrue(np.allclose(res.values, self.factor.values))


if __name__ == '__main__':
    unittest.main()
