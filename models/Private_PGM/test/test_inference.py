# This source code is licensed under the license found in the
# LICENSE file in the {root}/models/Private_PGM/ directory of this source tree.
import unittest
from models.Private_PGM.mbi.domain import Domain
from models.Private_PGM.mbi.inference import FactoredInference
from models.Private_PGM.mbi.graphical_model import CliqueVector
import numpy as np

class TestInference(unittest.TestCase):

    def setUp(self):
        attrs = ['a','b','c','d', 'e']
        shape = [2,3,4,5,6]
        self.domain = Domain(attrs, shape)

        #x = np.random.rand(*shape)

        self.measurements = []
        for i in range(4):
            I = np.eye(shape[i])
            y = np.random.rand(shape[i])
            y /= y.sum()
            self.measurements.append( (I, y, 1.0, attrs[i]) )

        self.engine = FactoredInference(self.domain, backend='numpy', log=True)

    def test_mirror_descent(self):
        loss = self.engine.mirror_descent(self.measurements, 1.0)
        self.assertEqual(self.engine.model.total, 1.0)
        self.assertTrue(loss <= 1e-4)

    def test_dual_averaging(self):
        loss = self.engine.dual_averaging(self.measurements, 1.0)
        self.assertEqual(self.engine.model.total, 1.0)
        #self.assertTrue(loss <= 1e-5)

    def test_interior_gradient(self):
        loss = self.engine.interior_gradient(self.measurements, 1.0)
        self.assertEqual(self.engine.model.total, 1.0)
        #self.assertTrue(loss <= 1e-5)

    def test_lipschitz(self):
        self.engine._setup(self.measurements, None)
        lip = self.engine._lipschitz(self.measurements)
        def rand():
            ans = {}
            for cl in self.engine.model.cliques:
                ans[cl] = self.engine.Factor.random(self.engine.domain.project(cl))
            return CliqueVector(ans)
        for _ in range(100):
            x = rand()
            y = rand()
            _, gx = self.engine._marginal_loss(x)
            _, gy = self.engine._marginal_loss(y)
            A = (gx-gy).dot(gx-gy)
            B = (x-y).dot(x-y)
            ratio = np.sqrt(A / B)
            self.assertTrue(ratio <= lip) 

if __name__ == '__main__':
    unittest.main()
