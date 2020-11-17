import numpy as np
import itertools
from collections import defaultdict
import copy
from mbi import *

#have to put asserts everywhere
class LBP():
    def __init__(self, domain, cliques, init_type = 'ones'):
        self.domain = domain
        self.cliques = cliques
        #assert that cliques have valid domain variables
        
        self.phi = LBP.createPhi(self.domain, self.cliques)
        self.mu_n = LBP.createVariableBeliefs(self.domain, self.cliques)
        self.mu_f = LBP.createFactorBeliefs(self.domain, self.cliques)
        
    #right now very vanilla. need to work in log space?
    @staticmethod
    def createPhi(domain, cliques):
        phi = {}
        for ind in cliques:
            phi[ind] = Factor.random(domain.project(ind))
        
        return phi
    
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
    
    @staticmethod
    def normalize(factor, norm_type):
        if norm_type == 'sum':
            return factor/factor.sum()
        elif norm_type == 'max':
            return factor/factor.max()
        else:
            print("Undefined Norm Type")
            return
    
    #norm_type, conv_criteria(?), tolerance
    def RunLBP(self, norm_type = 'sum', tol = 1e-4):
        
        count = 0
        while True:
            t_n = copy.deepcopy(self.mu_n)
            t_f = copy.deepcopy(self.mu_f)
            
            #variable to factor BP
            for v in self.domain:
                fac = [ind for ind in self.cliques if v in ind]
                for f in fac:
                    self.mu_n[v][f] = Factor.ones(self.domain.project(v))
                    comp = [a for a in fac if a is not f]
                    for c in comp:
                        self.mu_n[v][f] *= t_f[c][v] 
                    
                    #normalize
                    self.mu_n[v][f] = LBP.normalize(self.mu_n[v][f], norm_type)
            
            #factor to variable BP
            for ind in self.cliques:
                for v in ind:
                    comp = [a for a in ind if a is not v]
                    self.mu_f[ind][v] = copy.deepcopy(self.phi[ind])
                    for c in comp:
                        self.mu_f[ind][v] *= t_n[c][ind]
                    
                    self.mu_f[ind][v] = self.mu_f[ind][v].sum(comp)
                    #normalize
                    self.mu_f[ind][v] = LBP.normalize(self.mu_f[ind][v], norm_type)
            
            
            #check convergence
            f_t = []
            f_t1 = []
            for i in self.mu_f.keys():
                for j in self.mu_f[i].keys():
                    f_t.append(t_f[i][j].datavector())
                    f_t1.append(self.mu_f[i][j].datavector())
                    
            n_t = []
            n_t1 = []
            for i in self.mu_n.keys():
                for j in self.mu_n[i].keys():
                    n_t.append(t_n[i][j].datavector())
                    n_t1.append(self.mu_n[i][j].datavector())
                    
            n_t = np.concatenate(n_t).astype(None)
            n_t1 = np.concatenate(n_t1).astype(None)
            
            f_t = np.concatenate(f_t).astype(None)
            f_t1 = np.concatenate(f_t1).astype(None)
            
            if(np.allclose(f_t1,f_t,rtol=tol) and np.allclose(n_t1,n_t,rtol= tol)):
                print('Number of iterations: ',count)
                return 
            
            count += 1
            
    def marginals(self, marginal_vector):
        if len(marginal_vector) == 1:
            v = marginal_vector[0]
            p = Factor.ones(self.domain.project(v))
            # print(v, p.datavector())
            # print(self.cliques)
            fac = [ind for ind in self.cliques if v in ind]
            # print(fac)
            for f in fac:
                p *= self.mu_f[f][v]
            
            return LBP.normalize(p,'sum')
            
        else:
            #assert part of cliques? this will be required for this initialization.. will have to work out this thing
            #current assumpotion (wrong) is that marginal_vector is part of cliques
            #the for loop equation form of this will also change
            p = copy.deepcopy(self.phi[marginal_vector])
            # print(p)
            for v in marginal_vector:
                # print(v)
                p *= self.mu_n[v][marginal_vector]
                
            return LBP.normalize(p,'sum')
        
        
def main():
    var = ('A','B','C')
    sizes = (10,10,10)
    dom = Domain(var, sizes)
    indices = tuple(itertools.combinations(var,2))
    
    lbp = LBP(dom,indices)
    
    # print(lbp.mu_f)
    # print(lbp.mu_n)
    # for ind in indices:
    #     print(lbp.phi[ind].datavector())
    
    lbp.RunLBP()
    # p = lbp.marginals(('A','B'))
    p = lbp.marginals(('A',))
    print(p.datavector())
    
    prob = np.zeros(sizes)
    for a in range(sizes[0]):
        for b in range(sizes[1]):
            for c in range(sizes[2]):
                prob[a,b,c] = lbp.phi[('A','B')].values[a,b]*lbp.phi[('B','C')].values[b,c]*lbp.phi[('A','C')].values[a,c]
                # print(lbp.phi[('A','B')].values[a,b], lbp.phi[('B','C')].values[b,c], lbp.phi[('A','C')].values[a,c])
                
    pa = np.sum(prob,axis=(1,2))/np.sum(prob)
    p_ab = np.sum(prob,axis=2)/np.sum(prob)
    
    print(pa)
    # print(p_ab)
    
if __name__ == "__main__":
    main()