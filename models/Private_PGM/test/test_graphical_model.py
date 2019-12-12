# This source code is licensed under the license found in the
# LICENSE file in the {root}/models/Private_PGM/ directory of this source tree.
import unittest
from models.Private_PGM.mbi.domain import Domain
from models.Private_PGM.mbi.factor import Factor
from models.Private_PGM.mbi.graphical_model import GraphicalModel, CliqueVector
import numpy as np

class TestGraphicalModel(unittest.TestCase):

    def setUp(self):
        attrs = ['a','b','c','d']
        shape = [2,3,4,5]
        domain = Domain(attrs, shape)
        cliques = [('a','b'), ('b','c'),('c','d')]
        self.model = GraphicalModel(domain, cliques)
        zeros = { cl : Factor.zeros(domain.project(cl)) for cl in self.model.cliques }
        self.model.potentials = CliqueVector(zeros)

    def test_datavector(self):
        x = self.model.datavector()
        ans = np.ones(2*3*4*5) / (2*3*4*5)
        self.assertTrue(np.allclose(x, ans))

    def test_project(self):
        model = self.model.project(['d','a'])
        x = model.datavector()
        ans = np.ones(2*5) / 10.0
        self.assertEqual(x.size, 10)
        self.assertTrue(np.allclose(x, ans))

        model = self.model
        pot = { cl : Factor.random(model.domain.project(cl)) for cl in model.cliques }
        model.potentials = CliqueVector(pot)
        
        x = model.datavector(flatten=False)
        y0 = x.sum(axis=(2,3)).flatten()
        y1 = model.project(['a','b']).datavector() 
        self.assertEqual(y0.size, y1.size)
        self.assertTrue(np.allclose(y0, y1))

        x = model.project('a').datavector()

    def test_krondot(self):
        model = self.model
        pot = { cl : Factor.random(model.domain.project(cl)) for cl in model.cliques }
        model.potentials = CliqueVector(pot)
 
        A = np.ones((1,2))
        B = np.eye(3)
        C = np.ones((1,4))
        D = np.eye(5)
        res = model.krondot([A,B,C,D])
        x = model.datavector(flatten=False)
        ans = x.sum(axis=(0,2), keepdims=True)
        self.assertEqual(res.shape, ans.shape)
        self.assertTrue(np.allclose(res, ans))

    def test_calculate_many_marginals(self):
        proj = [[],['a'],['b'],['c'],['d'],['a','b'],['a','c'],['a','d'],['b','c'],
                ['b','d'],['c','d'],['a','b','c'],['a','b','d'],['a','c','d'],['b','c','d'],
                ['a','b','c','d']]
        proj = [tuple(p) for p in proj]
        model = self.model
        model.total = 10.0
        pot = { cl : Factor.random(model.domain.project(cl)) for cl in model.cliques }
        model.potentials = CliqueVector(pot)
        
        results = model.calculate_many_marginals(proj)
        for pr in proj:
            ans = model.project(pr).values
            close = np.allclose(results[pr].values, ans)
            print(pr, close, results[pr].values, ans)
            self.assertTrue(close)

    def test_belief_prop(self):
        pot = self.model.potentials
        self.model.total = 10
        mu = self.model.belief_propagation(pot)

        for key in mu:
            ans = self.model.total/np.prod(mu[key].domain.shape)
            self.assertTrue(np.allclose(mu[key].values, ans))

        pot = { cl : Factor.random(pot[cl].domain) for cl in pot }
        mu = self.model.belief_propagation(pot)

        logp = sum(pot.values())
        logp -= logp.logsumexp()
        dist = logp.exp() * self.model.total

        for key in mu:
            ans = dist.project(key).values  
            res = mu[key].values
            self.assertTrue(np.allclose(ans, res))

    def test_synthetic_data(self):
        model = self.model
        sy = model.synthetic_data()
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
