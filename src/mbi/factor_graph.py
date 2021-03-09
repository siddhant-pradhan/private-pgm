import numpy as np
from collections import defaultdict
from mbi import Domain, Factor
from mbi.graphical_model import CliqueVector

class FactorGraph():
    def __init__(self, domain, cliques, total = 1.0):
        self.domain = domain
        self.cliques = cliques
        self.total = total
        self.potentials = None
        self.marginals = None
        self.mu_f = None
        self.mu_n = None

    def datavector(self, flatten=True):
        """ Materialize the explicit representation of the distribution as a data vector. """
        logp = sum(self.potentials[cl] for cl in self.cliques)
        ans = np.exp(logp - logp.logsumexp())
        wgt = ans.domain.size() / self.domain.size()
        return ans.expand(self.domain).datavector(flatten) * wgt * self.total

    def init_messages(self):
        mu_n = defaultdict(dict)
        mu_f = defaultdict(dict)
        for cl in self.cliques:
            for v in cl:
                mu_f[cl][v] = Factor.zeros(self.domain.project(v))
                mu_n[v][cl] = Factor.zeros(self.domain.project(v))
        return mu_n, mu_f
    
    def project(self,attrs):
        if type(attrs) is list:
            attrs = tuple(attrs)
        
        if self.marginals is not None:
            for cl in self.cliques:
                if set(attrs) <= set(cl):
                    return self.marginals[cl].project(attrs)
        
        if self.mu_n is None:
            self.loopy_belief_propagation(self.potentials, iters=10)

        mu_n, mu_f = self.mu_n, self.mu_f
        msgs = {} 
        for v in attrs:
            fac = [cl for cl in self.cliques if v in cl]
            msgs[v] = Factor.zeros(self.domain.project(v)) + sum(mu_f[f][v] for f in fac)
                    
        p =  Factor.zeros(self.domain.project(attrs)) + sum(msgs[v] for v in attrs)
        p += np.log(self.total) - p.logsumexp()
        return p.exp()
           
    def belief_propagation(self, potentials):
        return self.loopy_belief_propagation(potentials, iters=10)
    
    def loopy_belief_propagation(self, potentials, iters = 100, callback=None):
       
        mu_n, mu_f = self.init_messages() 
        self.potentials = potentials
        
        for i in range(iters):
            #variable to factor BP
            for v in self.domain:
                fac = [cl for cl in self.cliques if v in cl]
                for f in fac:
                    complement = [var for var in fac if var is not f]
                    mu_n[v][f] = Factor.zeros(self.domain.project(v))
                    for c in complement:
                        mu_n[v][f] += mu_f[c][v]
                    mu_n[v][f] -= mu_n[v][f].logsumexp()
            
            #factor to variable BP
            for cl in self.cliques:
                for v in cl:
                    complement = [var for var in cl if var is not v]
                    mu_f[cl][v] = potentials[cl] + sum(mu_n[c][cl] for c in complement)
                    for c in complement:
                        mu_f[cl][v] += mu_n[c][cl]

                    mu_f[cl][v] = mu_f[cl][v].logsumexp(complement)
                    
                    #normalize
                    mu_f[cl][v] -= mu_f[cl][v].logsumexp()
            
            if callback is not None:
                mg = self.clique_marginals(mu_n, mu_f, potentials)
                callback(mg)
            
        self.mu_f, self.mu_n = mu_f, mu_n
        return  self.clique_marginals(mu_n, mu_f, potentials)

    def convergent_belief_propagation(self, potentials, counting_v = None, iters=100, callback=None):
        # Algorithm 11.2 in Koller & Friedman (modified to work in log space)

        counting_vhat = {}
        for i in self.domain:
            nbrs = [r for r in self.cliques if i in r]
            counting_vhat[i] = counting_v[i] + sum(counting_v[r] for r in nbrs)
            for r in nbrs:
                counting_vhat[i,r] = counting_v[r] + counting_v[i,r]

        mu_n = defaultdict(dict)
        mu_f = defaultdict(dict)
        for r in self.cliques:
            for j in r:
                mu_f[r][j] = Factor.zeros(self.domain.project(j))
                mu_n[j][r] = Factor.zeros(self.domain.project(r))
        #mu_n, mu_f = self.init_messages()

        for _ in range(iters):
            for i in self.domain:
                nbrs = [r for r in self.cliques if i in r]
                for r in nbrs:
                    comp = [j for j in r if i != j]
                    mu_f[r][i] = potentials[r] + sum(mu_n[j][r] for j in comp)
                    mu_f[r][i] /= counting_vhat[i,r]
                    mu_f[r][i] = mu_f[r][i].logsumexp(comp)
                belief = Factor.zeros(self.domain.project(i)) + sum(mu_f[r][i]*counting_vhat[i,r] for r in nbrs) / counting_vhat[i]
                belief -= belief.logsumexp()
                for r in nbrs:
                    comp = [j for j in r if i != j]
                    A = -counting_v[i,r]/counting_vhat[i,r]
                    B = counting_v[r]
                    mu_n[i][r] = A*(potentials[r] + sum(mu_n[j][r] for j in comp))
                    mu_n[i][r] += B*(belief - mu_f[r][i])
            if callback is not None:
                mg = self.clique_marginals(mu_n, mu_f, potentials)
                callback(mg)

        return self.clique_marginals(mu_n, mu_f, potentials)

    def clique_marginals(self, mu_n, mu_f, potentials):
        marginals = {}
        for cl in self.cliques:
            belief = potentials[cl] + sum(mu_n[v][cl] for v in cl)
            belief += np.log(self.total) - belief.logsumexp()
            marginals[cl] = belief.exp()
        return CliqueVector(marginals)


    def get_counting_numbers(self):
        index = {}
        idx = 0

        for i in self.domain:
            index[i] = idx
            idx += 1
        for r in self.cliques:
            index[r] = idx
            idx += 1
            
        for r in self.cliques:
            for i in r:
                index[r,i] = idx
                idx += 1
                    
        vectors = {}
        for r in self.cliques:
            v = np.zeros(idx)
            v[index[r]] = 1
            for i in r:
                v[index[r,i]] = 1
            vectors[r] = v
                

        for i in self.domain:
            v = np.zeros(idx)
            v[index[i]] = 1
            for r in self.cliques:
                if i in r:
                    v[index[r,i]] = -1
            vectors[i] = v
            
        constraints = []
        for i in self.domain:
            con = vectors[i].copy()
            for r in self.cliques:
                if i in r:
                    con += vectors[r]
            constraints.append(con)
        A = np.array(constraints)
        b = np.ones(len(self.domain))

        X = np.vstack([vectors[r] for r in self.cliques])
        y = np.ones(len(self.cliques))
        P = X.T @ X
        q = -X.T @ y
        G = -np.eye(q.size)
        h = np.zeros(q.size)

        from cvxopt import solvers, matrix

        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)

        ans = solvers.qp(P, q, G, h, A, b)
        x = np.array(ans['x']).flatten()
        #for p in vectors: print(p, vectors[p] @ x)

        counting_v = {}
        for r in self.cliques:
            counting_v[r] = x[index[r]]
            for i in r:
                counting_v[i,r] = x[index[r,i]]
        for i in self.domain:
            counting_v[i] = x[index[i]]
        return counting_v


if __name__ == '__main__':
    import itertools
    var = ['A','B','C','D','E']
    dom = Domain(var, [2,3,4,5,6])
    cliques = list(itertools.combinations(var, 2)) + list(itertools.combinations(var, 3)) + list(itertools.combinations(var, 4))
    cliques = [cliques[i] for i in np.random.choice(len(cliques), 10, replace=False)]

    potentials = {}
    for cl in cliques:
        potentials[cl] = Factor.zeros(dom.project(cl))

    model = FactorGraph(dom, cliques)
    counting_v = model.get_counting_numbers()
    marginals = model.convergent_belief_propagation(potentials, counting_v=counting_v)
    for cl in cliques:
        print(cl, marginals[cl].datavector())    

