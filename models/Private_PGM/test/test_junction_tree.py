# This source code is licensed under the license found in the
# LICENSE file in the {root}/models/Private_PGM/ directory of this source tree.
import unittest
from models.Private_PGM.mbi.domain import Domain
from models.Private_PGM.mbi.junction_tree import JunctionTree

class TestJunctionTree(unittest.TestCase):

    def setUp(self):
        attrs = ['a','b','c','d']
        shape = [10,20,30,40]
        domain = Domain(attrs, shape)
        cliques = [('a','b'), ('b','c'),('c','d')]
        self.tree = JunctionTree(domain, cliques)

    def test_maximal_cliques(self):
        ans = [set(x) for x in [('a','b'), ('b','c'),('c','d')]]
        res = self.tree.maximal_cliques()
        for cl in res:
            self.assertTrue(set(cl) in ans)

    def test_mp_order(self):
        order = self.tree.mp_order()
        self.assertEqual(len(order), 4)
        print(order)

    def test_separator_axes(self):
        res = self.tree.separator_axes()
        ans = { 'b', 'c' } 
        res = set.union(*map(set, res.values()))
        self.assertEqual(res, ans)
        
    def test_neighbors(self):
        res = self.tree.neighbors()    

if __name__ == '__main__':
    unittest.main()
