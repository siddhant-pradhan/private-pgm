import unittest
from mbi import Domain, Factor, ApproxGraphicalModel
from mbi.graphical_model import CliqueVector
import numpy as np

class TestGraphicalModel(unittest.TestCase):

    def setUp(self):

        var = ('A','B','C')
        sizes = (2,3,4)
        domain = Domain(var, sizes)
        cliques = [('A','B'), ('B','C')]
        
        self.model = ApproxGraphicalModel(domain, cliques)
        zeros = { cl : Factor.zeros(domain.project(cl)) for cl in self.model.cliques }
        self.model.potentials = CliqueVector(zeros)

    def test_datavector(self):
        x = self.model.datavector()
        ans = np.ones(2*3*4) / (2*3*4)
        self.assertTrue(np.allclose(x, ans))

    def test_project(self):
        model = self.model.project(['A','C'])
        x = model.datavector()
        ans = np.ones(2*4) / 8.0
        self.assertEqual(x.size, 8)
        self.assertTrue(np.allclose(x, ans))

        model = self.model
        pot = { cl : Factor.random(model.domain.project(cl)) for cl in model.cliques }
        model.potentials = CliqueVector(pot)
       
        model.loopy_belief_propagation(model.potentials, 100)  
        x = model.datavector(flatten=False)
        y0 = x.sum(axis=2).flatten()
        y1 = model.project(['A','B']).datavector() 
        self.assertEqual(y0.size, y1.size)
        print(y0, y1)
        self.assertTrue(np.linalg.norm(y0 -y1, 1) <= 0.05)

        x = model.project('A').datavector()

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
            self.assertTrue(np.linalg.norm(ans-res,1) <= 0.05)

if __name__ == '__main__':
    unittest.main()
