import numpy as np
import itertools
from collections import defaultdict
import copy
from mbi import *
from mbi.graphical_model import CliqueVector
import sys
import random

np.random.seed(0)

class LBP():
    def __init__(self, domain, cliques, init_type = 'ones', total = 1.0):
        self.domain = domain
        self.cliques = cliques
        self.total = total
        
        self.full_list = self.cliques
        #changed here to only cliques. marginal convergence may run into issues
        # for i in self.domain.attrs:
        #     self.full_list = self.full_list + tuple(i)
        
        # self.phi = LBP.createPhi(self.domain, self.cliques)
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

    @staticmethod
    def createPhi(domain, cliques):
        phi = CliqueVector({ cl : Factor.random(domain.project(cl)).log() for cl in cliques })
        
        return phi
    
    @staticmethod
    def createVariableBeliefs(domain, cliques, init_type = 'zeros'):
        mu_n = defaultdict(dict)
        for v in domain:
            fac = [cl for cl in cliques if v in cl]
            for f in fac:
                if init_type == 'ones':
                    mu_n[v][f] = Factor.ones(domain.project(v))
                elif init_type == 'zeros':
                    mu_n[v][f] = Factor.zeros(domain.project(v))
                elif init_type == 'random':
                    mu_n[v][f] = Factor.random(domain.project(v))
        
        return mu_n
    
    @staticmethod
    def createFactorBeliefs(domain, cliques, init_type = 'zeros'):
        mu_f = defaultdict(dict)
        for cl in cliques:
            for v in cl:
                if init_type == 'ones':
                    mu_f[cl][v] = Factor.ones(domain.project(v))
                elif init_type == 'zeros':
                    mu_f[cl][v] = Factor.zeros(domain.project(v))
                elif init_type == 'random':
                    mu_f[cl][v] = Factor.random(domain.project(v))
                    
        return mu_f
    
    @staticmethod
    def normalize(factor, norm_type):
        # factor = factor.exp()
        if norm_type == 'sum':
            return factor/factor.sum()
        elif norm_type == 'max':
            return factor/factor.max()
        else:
            print("Undefined Norm Type")
            return
    
    def getPhi(self):
        return self.phi
    
    @staticmethod
    def norm(arr1, arr2, norm_type = 'L1'):
        assert isinstance(arr1,np.ndarray)
        if norm_type == 'L1':
            return np.sum(np.abs(arr1-arr2))/arr1.size
    
    @staticmethod    
    def check_convergence(obj_t, obj_t1, conv_type = 'L1', conv_tol = 1e-4, nested_dict = False):
        arr_t = []
        arr_t1 = []
        count = 0
        if nested_dict:
            for i in obj_t.keys():
                for j in obj_t[i].keys():
                    arr_t.append(obj_t[i][j].datavector())
                    arr_t1.append(obj_t1[i][j].datavector())
                    count += 1
        
        else:
            
            for i in obj_t.keys():
                arr_t.append(obj_t[i].datavector())
                arr_t1.append(obj_t1[i].datavector())
                count += 1
                
        arr_t = np.concatenate(arr_t).astype(None)
        arr_t1 = np.concatenate(arr_t1).astype(None)
        
        if conv_type == 'L1':
            return np.sum(np.abs(arr_t-arr_t1))/count <= conv_tol
        
        elif conv_type == 'Diff':
            return np.allclose(arr_t,arr_t1,atol=conv_tol)
        
    
    def project(self,attrs):
        
        if type(attrs) is list:
            attrs = tuple(attrs)
        
        if self.marginals is not None:
            for cl in self.cliques:
                if set(attrs) <= set(cl):
                    return self.marginals[cl].project(attrs)
                
        #get domain to project proper domains
        #insert key in potentials
        #add to clique
        #do LBP
        #return marginal
        
        proj_domain = self.domain.project(attrs)
        #avoid this
        # self.cliques.append(attrs)
        # self.potentials[attrs] = Factor.zeros(proj_domain)
        ##########################################################
        
        # new_marginals = self.loopy_belief_propagation(self.potentials)
        # return new_marginals[attrs]        
        
        ##########################################################   
        # if self.mu_f is None and self.mu_f is None:
        #     # print('hi')
        #     new_marginals = self.loopy_belief_propagation(self.potentials,num_iter=10)
        #     return new_marginals[attrs]
        
        #######################################################
        
        mu_n = copy.deepcopy(self.mu_n)
        mu_f = copy.deepcopy(self.mu_f)
        
        
        #variable to factor BP
        for v in attrs:
            fac = [cl for cl in self.cliques if v in cl]
            # fac.append(attrs)
            mu_n[v][attrs] = Factor.zeros(self.domain.project(v))
            for f in fac:
                mu_n[v][attrs] += mu_f[f][v]
            
            mu_n[v][attrs] -= mu_n[v][attrs].logsumexp() 
            
            # for f in fac:
            #     mu_n[v][f] = Factor.zeros(self.domain.project(v))
            #     complement = [cl for cl in fac if cl is not f]
                
            #     for c in complement:
            #         #updates modified, they are not traditional LBP
            #         # mu_n[v][f] += t_f[c][v] 
            #         mu_n[v][f] += mu_f[c][v]
            #         # print(v,f,c)
                
            #     #normalize
            #     # mu_n[v][f] = LBP.normalize(mu_n[v][f], norm_type)
            #     mu_n[v][f] -= mu_n[v][f].logsumexp()   
                    
        p =  Factor.zeros(proj_domain)  
        for v in attrs:
            p += mu_n[v][attrs]
            
        p += np.log(self.total) - p.logsumexp()
        # return LBP.normalize(p.exp(),'sum')
        return p.exp()
            
        # return CliqueVector(self.single_marginal(attrs,self.mu_n,self.mu_f,self.potentials))
            
    
    def loopy_belief_propagation(self, potentials, norm_type = 'sum', tol = 1e-4, conv_type = 'iterations', num_iter = 100, conv_crit = 'L1', alpha = 1.0, callback = None):
        
        # print(self.cliques)
        mu_n = None
        mu_f = None
        # if self.mu_n is None and self.mu_f is None:
        #     mu_n = LBP.createVariableBeliefs(self.domain, self.cliques)
        #     mu_f = LBP.createFactorBeliefs(self.domain, self.cliques)
        # else:
        #     mu_n = copy.deepcopy(self.mu_n)
        #     mu_f = copy.deepcopy(self.mu_f)
            
        mu_n = LBP.createVariableBeliefs(self.domain, self.cliques)
        mu_f = LBP.createFactorBeliefs(self.domain, self.cliques)
        
        self.potentials = potentials
        
        total_iterations = 0
        if conv_type == 'messages' or conv_type == 'marginals':
            total_iterations = sys.maxsize
        elif conv_type == 'iterations':
            total_iterations = num_iter
            
        for i in range(total_iterations):
            # print(i)
            t_n = copy.deepcopy(mu_n)
            t_f = copy.deepcopy(mu_f)
            
            #variable to factor BP
            for v in self.domain:
                fac = [cl for cl in self.cliques if v in cl]
                for f in fac:
                    mu_n[v][f] = Factor.zeros(self.domain.project(v))
                    complement = [var for var in fac if var is not f]
                    
                    for c in complement:
                        #updates modified, they are not traditional LBP
                        # mu_n[v][f] += t_f[c][v] 
                        mu_n[v][f] += mu_f[c][v]
                    
                    #normalize
                    
                    mu_n[v][f] -= mu_n[v][f].logsumexp()
                    #damped equation
                    mu_n[v][f] = alpha*mu_n[v][f] + (1-alpha)*t_n[v][f]
            #factor to variable BP
            for cl in self.cliques:
                for v in cl:
                    complement = [var for var in cl if var is not v]
                    mu_f[cl][v] = copy.deepcopy(potentials[cl])
                    for c in complement:
                        #updates modified, they are not traditional LBP
                        # mu_f[cl][v] += t_n[c][cl]
                        mu_f[cl][v] += mu_n[c][cl]

                    mu_f[cl][v] = mu_f[cl][v].logsumexp(complement)
                    
                    #normalize
                    # mu_f[cl][v] = LBP.normalize(mu_f[cl][v], norm_type)
                    mu_f[cl][v] -= mu_f[cl][v].logsumexp()
                    #damped equation
                    mu_f[cl][v] = alpha*mu_f[cl][v] + (1-alpha)*t_f[cl][v]
                    
            #comment and delete   
            # mg = self.all_marginals(t_n,t_f, potentials)
            # mg1 = self.all_marginals(mu_n, mu_f, potentials)
            # l1 = 0
            # count = 0
            # for k in mg.keys():
            #     l1 += np.sum(np.abs(mg[k].datavector() - mg1[k].datavector()))
            #     count += 1
                
            # print(l1/count)
            
            if conv_type == 'messages':
                truth = LBP.check_convergence(t_f,mu_f,conv_crit,tol) and LBP.check_convergence(t_n,mu_n,conv_crit,tol)
                if truth:
                    self.mu_f, self.mu_n = copy.deepcopy(mu_f), copy.deepcopy(mu_n)
                    return self.all_marginals(mu_n,mu_f, potentials)
                
            if callback is not None:
                mg = self.all_marginals(mu_n, mu_f, potentials)
                callback(mg)
            
            elif conv_type == 'marginals' and i != 0:
                mg = self.all_marginals(t_n,t_f, potentials)
                mg1 = self.all_marginals(mu_n, mu_f, potentials)
                truth = LBP.check_convergence(mg,mg1,conv_crit,tol)
                if truth:
                    self.mu_f, self.mu_n = copy.deepcopy(mu_f), copy.deepcopy(mu_n)
                    # print(i)
                    #change the return type
                    return i+1, self.all_marginals(mu_n,mu_f, potentials)
            
        self.mu_f, self.mu_n = copy.deepcopy(mu_f), copy.deepcopy(mu_n)
        return  self.all_marginals(mu_n,mu_f, potentials)
       
    def asynchronous_lbp(self, potentials, tol = 1e-4, conv_type = 'iterations', num_iter = 100, conv_crit = 'L1', alpha = 1.0, callback = None, randomize = False):
        
        mu_n = None
        mu_f = None
        # if self.mu_n is None and self.mu_f is None:
        #     mu_n = LBP.createVariableBeliefs(self.domain, self.cliques)
        #     mu_f = LBP.createFactorBeliefs(self.domain, self.cliques)
        # else:
        #     mu_n = copy.deepcopy(self.mu_n)
        #     mu_f = copy.deepcopy(self.mu_f)
        
        variables = copy.deepcopy(self.domain)
        if randomize:
            random.shuffle(variables)
            
        mu_n = LBP.createVariableBeliefs(self.domain, self.cliques)
        mu_f = LBP.createFactorBeliefs(self.domain, self.cliques)
        
        self.potentials = potentials
        
        total_iterations = 0
        if conv_type == 'messages' or conv_type == 'marginals':
            total_iterations = sys.maxsize
        elif conv_type == 'iterations':
            total_iterations = num_iter
            
        for i in range(total_iterations):
            t_n = copy.deepcopy(mu_n)
            t_f = copy.deepcopy(mu_f)
            
            for v in variables:
                fac = [cl for cl in self.cliques if v in cl]
                for f in fac:
                    mu_n[v][f] = Factor.zeros(self.domain.project(v))
                    complement_factors = [c_f for c_f in fac if c_f is not f]
                    
                    for cf in complement_factors:
                        #updates modified, they are not traditional LBP
                        # mu_n[v][f] += t_f[c][v] 
                        mu_n[v][f] += mu_f[cf][v]
                    
                    #normalize
                    
                    mu_n[v][f] -= mu_n[v][f].logsumexp()
                    #damped equation
                    mu_n[v][f] = alpha*mu_n[v][f] + (1-alpha)*t_n[v][f]
                    
                    
                    ##################################################################
                    complement_variables = [var for var in f if var is not v]
                    mu_f[f][v] = copy.deepcopy(potentials[f])
                    for cv in complement_variables:
                        #updates modified, they are not traditional LBP
                        # mu_f[cl][v] += t_n[c][cl]
                        mu_f[f][v] += mu_n[cv][f]

                    mu_f[f][v] = mu_f[f][v].logsumexp(complement_variables)
                    
                    #normalize
                    # mu_f[cl][v] = LBP.normalize(mu_f[cl][v], norm_type)
                    mu_f[f][v] -= mu_f[f][v].logsumexp()
                    #damped equation
                    mu_f[f][v] = alpha*mu_f[f][v] + (1-alpha)*t_f[f][v]
                    
            
            #
            if conv_type == 'messages':
                truth = LBP.check_convergence(t_f,mu_f,conv_crit,tol) and LBP.check_convergence(t_n,mu_n,conv_crit,tol)
                if truth:
                    self.mu_f, self.mu_n = copy.deepcopy(mu_f), copy.deepcopy(mu_n)
                    return self.all_marginals(mu_n,mu_f, potentials)
                
            if callback is not None:
                mg = self.all_marginals(mu_n, mu_f, potentials)
                callback(mg)
            
            elif conv_type == 'marginals' and i != 0:
                mg = self.all_marginals(t_n,t_f, potentials)
                mg1 = self.all_marginals(mu_n, mu_f, potentials)
                truth = LBP.check_convergence(mg,mg1,conv_crit,tol)
                if truth:
                    self.mu_f, self.mu_n = copy.deepcopy(mu_f), copy.deepcopy(mu_n)
                    # print(i)
                    #change the return type
                    return i+1, self.all_marginals(mu_n,mu_f, potentials)
            
        self.mu_f, self.mu_n = copy.deepcopy(mu_f), copy.deepcopy(mu_n)
        return  self.all_marginals(mu_n,mu_f, potentials)
                    
                    
                    
              
            
    def single_marginal(self, marginal_vector, mu_n, mu_f, potentials):
        if len(marginal_vector) == 1:
            v = marginal_vector[0]
            p = Factor.zeros(self.domain.project(v))
            fac = [cl for cl in self.cliques if v in cl]
            
            for f in fac:
                p += mu_f[f][v]

            p += np.log(self.total) - p.logsumexp()
            # return LBP.normalize(p.exp(),'sum')
            return p.exp()
            
        else:
            p = copy.deepcopy(potentials[marginal_vector])
            
            for v in marginal_vector:
                p += mu_n[v][marginal_vector]
                
            p += np.log(self.total) - p.logsumexp()
            # return LBP.normalize(p.exp(),'sum')
            return p.exp()
        
    def all_marginals(self, mu_n, mu_f, potentials):
        
        all_marginals = {}
        #changed here to only cliques. marginal convergence may run into issues
        for c in self.cliques:
            all_marginals[c] = self.single_marginal(c, mu_n, mu_f, potentials)
            
        
        self.marginals = CliqueVector(all_marginals)    
        return CliqueVector(all_marginals)
    
        # return CliqueVector({cl : self.marginals(cl, mu_n, mu_f)} for cl in self.full_list)
        
        
def main():
    # var = ('A','B','C')
    # sizes = (10,10,10)
    # dom = Domain(var, sizes)
    # indices = tuple(itertools.combinations(var,2))
    
    # lbp = LBP(dom,indices)
    

    # counts, m = lbp.loopy_belief_propagation(lbp.phi, conv_type='marginals')

    # print(m[('A','B')].datavector())
    # print(m[('A')].datavector())
    
    ##################################################################################
    # n=20
    # dom_size = 10
    # var = []
    # sizes = []
    # for i in range(n):
    #     var.append(str(i))
    #     sizes.append(dom_size)
        
    # var = tuple(var)
    # sizes = tuple(sizes)
    # dom = Domain(var, sizes)
    # indices = tuple(itertools.combinations(var,2))
    
    # lbp = LBP(dom,indices)
    
    # counts, m = lbp.loopy_belief_propagation(lbp.phi)
    
    # print(counts)
    var = ('A','B','C')
    sizes = (2,3,4)
    dom = Domain(var, sizes)
    cliques = [('A','B'), ('B','C')]
    lbp = LBP(dom, cliques)
    potentials = { cl : Factor.zeros(dom.project(cl)) for cl in cliques }
    marginals = lbp.loopy_belief_propagation(potentials, num_iter=100)
    print(marginals[('A','B')].datavector())
    
    
if __name__ == "__main__":
    main()