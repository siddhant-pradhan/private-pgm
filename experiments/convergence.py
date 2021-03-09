from mbi import Domain, Factor, ApproxGraphicalModel, FactorGraph
from mbi.graphical_model import CliqueVector
import itertools
import numpy as np
import matplotlib.pyplot as plt

def exact_marginals(model, cliques):
    M = Factor(model.domain, model.datavector())
    return CliqueVector({ cl : M.project(cl) for cl in cliques })

def error(true, est):
    diff = true - est
    return np.mean([0.5*np.linalg.norm(d.datavector(), 1) for d in diff.values()])

if __name__ == '__main__':

    mult_factor = 1
    iters = 1000
    trials = 5
    K = 3


    attributes = ['A','B','C']#,'D','E','F','G']
    sizes = [8 for _ in attributes]
    domain = Domain(attributes, sizes)

    pairs = list(itertools.combinations(attributes, 2))


    for i in range(trials):
        ys = []
        idx = np.random.choice(len(pairs), K, replace=False)
        cliques = [pairs[i] for i in idx]

        model = FactorGraph(domain, cliques)
        potentials = { cl : Factor.random(domain.project(cl))*mult_factor for cl in cliques }
        model.potentials = potentials
        true = exact_marginals(model, cliques)

        def callback(est):
            E = error(true, est)
            ys.append(E)
       
        counting_v = model.get_counting_numbers() 
        est = model.convergent_belief_propagation(potentials, counting_v=counting_v, iters=iters, callback=callback)
        plt.plot(range(iters), ys, '.', label='Loopy BP')


    plt.title('7D domain of size (8, 8, ..., 8)')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Distance to True Marginals')
    plt.savefig('convergence_%d_%d.png' % (K, mult_factor))
    plt.show()

      

