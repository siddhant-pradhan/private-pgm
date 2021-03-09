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
    iters = 100
    trials = 5


    attributes = ['A','B','C','D','E','F','G']
    sizes = [8 for _ in attributes]
    domain = Domain(attributes, sizes)

    pairs = list(itertools.combinations(attributes, 2))

    xs = []
    ys = []
    zs = []

    for K in range(1, len(pairs)+1):
        for i in range(trials):
            idx = np.random.choice(len(pairs), K, replace=False)
            
            cliques = [pairs[i] for i in idx]

            model = FactorGraph(domain, cliques)
            potentials = { cl : Factor.random(domain.project(cl))*mult_factor for cl in cliques }
           
            counting_v = model.get_counting_numbers()
            est = model.convergent_belief_propagation(potentials, counting_v=counting_v, iters=iters)
            model.potentials = potentials
            true = exact_marginals(model, cliques)
            unif = CliqueVector({ cl : Factor.uniform(domain.project(cl)) for cl in cliques })
            E = error(true, est)
            E2 = error(true, unif)
            xs.append(K)
            ys.append(E)
            zs.append(E2)

            print(K, E)

    plt.plot(xs, ys, '.', label='Loopy BP')
    plt.plot(xs, zs, '.', label='Uniform')
    plt.title('7D domain of size (8, 8, ..., 8)')
    plt.legend()
    plt.xlabel('Number of Edges')
    plt.ylabel('Average Variation Distance')
    plt.savefig('graph_density_%d_%d_%d.png' % (mult_factor, iters, trials))
    plt.show()

      

