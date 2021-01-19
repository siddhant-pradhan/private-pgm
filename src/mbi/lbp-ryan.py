import numpy as np
import itertools
from collections import defaultdict
import copy
from mbi import *
import time
from mbi.graphical_model import CliqueVector

#have to put asserts everywhere
class LBP():
    def __init__(self, domain, cliques, init_type = 'ones'):
        self.domain = domain
        self.cliques = cliques
        #assert that cliques have valid domain variables
        
        self.mu_n = LBP.createVariableBeliefs(self.domain, self.cliques)
        self.mu_f = LBP.createFactorBeliefs(self.domain, self.cliques)
        
    @staticmethod
    def createVariableBeliefs(domain, cliques, init_type = 'ones'):
        mu_n = defaultdict(dict)
        for v in domain:
            fac = [ind for ind in cliques if v in ind]
            for f in fac:
                if init_type == 'ones':
                    mu_n[v][f] = Factor.ones(domain.project(v))
                elif init_type == 'random':
                    mu_n[v][f] = Factor.random(domain.project(v))
        
        return mu_n
    
    @staticmethod
    def createFactorBeliefs(domain, cliques, init_type = 'ones'):
        mu_f = defaultdict(dict)
        for ind in cliques:
            for v in ind:
                if init_type == 'ones':
                    mu_f[ind][v] = Factor.ones(domain.project(v))
                elif init_type == 'ones':
                    mu_f[ind][v] = Factor.random(domain.project(v))
                    
        return mu_f
    
    def loopy_belief_propagation(self, potentials, iters=100):
        beliefs = {}
        for cl in self.cliques:
            beliefs[cl] = potentials[cl].copy()
            for v in cl:
                beliefs[cl] += self.mu_n[v][cl]
            beliefs[cl] -= beliefs[cl].logsumexp()
        marginals = CliqueVector(beliefs).exp()
 
  
        for _ in range(iters):
            #variable to factor BP
            for v in self.domain:
                fac = [ind for ind in self.cliques if v in ind]
                for f in fac:
                    self.mu_n[v][f] = Factor.zeros(self.domain.project(v))
                    comp = [a for a in fac if a is not f]
                    for c in comp:
                        self.mu_n[v][f] += self.mu_f[c][v] 
            
            #factor to variable BP
            for ind in self.cliques:
                for v in ind:
                    comp = [a for a in ind if a is not v]
                    self.mu_f[ind][v] = copy.deepcopy(potentials[ind])
                    for c in comp:
                        self.mu_f[ind][v] += self.mu_n[c][ind]
                    
                    self.mu_f[ind][v] = self.mu_f[ind][v].logsumexp(comp)


            old_marginals = marginals
            beliefs = {}
            for cl in self.cliques:
                beliefs[cl] = potentials[cl].copy()
                for v in cl:
                    beliefs[cl] += self.mu_n[v][cl]
                beliefs[cl] -= beliefs[cl].logsumexp()
            marginals = CliqueVector(beliefs).exp()
     
            print(marginals[('A','B')].project('A').datavector())
            #diff = marginals - old_marginals
            #print(diff.dot(diff))
            #print(np.concatenate([marginals[cl].datavector() for cl in self.cliques]))

        return marginals
 

def main():
    import string
    d = 3
    var = tuple(string.ascii_uppercase[:d])
    sizes = tuple([4]*d)
    dom = Domain(var, sizes)
    cliques = tuple(itertools.combinations(var,2))
    
    lbp = LBP(dom,cliques)
    
    # print(lbp.mu_f)
    # print(lbp.mu_n)
    # for ind in indices:
    #     print(lbp.phi[ind].datavector())
  
    np.random.seed(14) 
    potentials = CliqueVector({ cl : Factor.random(dom.project(cl)).log() for cl in cliques })
 
    marginals = lbp.loopy_belief_propagation(potentials, iters=10)
    # p = lbp.marginals(('A','B'))
    print(marginals[('A','B')].datavector())
   
    if d <= 5: 
        phi = potentials.exp()
        ans = Factor.ones(dom)
        for cl in lbp.cliques:
            ans *= phi[cl]
        ans /= ans.sum()
        print(ans.project(('A','B')).datavector())
    

    
if __name__ == "__main__":
    main()
