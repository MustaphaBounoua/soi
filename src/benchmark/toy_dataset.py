
import numpy as np
import torch
from numpy.random import default_rng
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset
import itertools
from itertools import chain, combinations
from sklearn import preprocessing
from scipy.special import erf

N = 100000

def eye(ma,dim):
    if isinstance(ma, (list, tuple, np.ndarray)):
        new_m = np.zeros((ma.shape[0],ma.shape[0],dim,dim))
        for i in range (ma.shape[0]):
            for j in range (ma.shape[0]):
                new_m[i][j] =  ma[i][j] * np.eye(dim)
        return np.transpose(new_m, (0,2,1,3)).reshape((ma.shape[0]*dim,ma.shape[0]*dim ))
    else:
        return ma * np.eye(dim)
        

def entropy(cov,dim, use_det =False):

    if use_det:
        
        if isinstance(cov, (list, tuple, np.ndarray)) or dim>1:
            cov_d = eye(cov,dim)
            D = cov_d.shape[0] 
            e = 0.5 * D * (1 + np.log(2 * np.pi) ) + 0.5* np.log(np.linalg.det(cov_d) ) 
            return e
        else:
            D = 1 
            e = 0.5 * D * (1 + np.log(2 * np.pi) ) + 0.5* np.log(cov ) 
            return e
            
    else:
        dist = multivariate_normal(mean=None, cov=cov)
        return   dim * dist.entropy()    

# def get_cov_minus_i(cov,i):
#     cov_list= cov.tolist()
#     cov_list.pop(i)
#     cov_list =np.array( cov_list).T.tolist()
#     cov_list.pop(i)
#     return np.array(cov_list).T


def get_cov_minus_i(cov,i):
    i.sort()
    k=0
    for j in i:
        j = j-k
        cov_list= cov.tolist()
        cov_list.pop(j)
        cov_list =np.array( cov_list).T.tolist()
        cov_list.pop(j)
        cov = np.array(cov_list).T
        k+=1
    return cov



def tc(cov,dim,use_det):

    nb_var = cov.shape[0] 
    return np.sum( [ entropy(cov[i][i],dim,use_det ) for i in range(nb_var) ] ) - entropy(cov,dim,use_det) 


def o_inf(cov,dim,use_det ):
        nb_var = cov.shape[0] 
        tc_i =[ tc(get_cov_minus_i(cov,[i]),dim,use_det) for i in range(nb_var )] 
        return (2-nb_var) * tc(cov,dim,use_det) + np.sum(tc_i) , tc_i
    
def dtc(cov,dim,use_det):
        nb_var = cov.shape[0] 
        return  (nb_var-1) * tc(cov,dim,use_det) - np.sum(
        [ tc( get_cov_minus_i(cov,[i])  ,dim,use_det)
            for i in range(cov.shape[0] )
        ] )


def s_inf(cov,dim,use_det):
        nb_var = cov.shape[0] 
        return  (nb_var) * tc(cov,dim,use_det) - np.sum(
        [ tc( get_cov_minus_i(cov,[i]),dim ,use_det )
            for i in range(cov.shape[0] )
        ] )




def get_powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class Task():
    def __init__(self, nb_var,sigma,dim ,use_det =False,transformation=None) :
        self.nb_var = nb_var
        self.sigma = sigma
        self.cov = None
        self.dim = dim
        self.use_det = use_det
        self.transformation = transformation
    
    def get_summary(self):
        return{ "tc" : self.tc() ,
               "dtc": self.dtc() , 
               "o_inf": self.o_inf()[0] , 
               "s_inf": self.s_inf(), 
               "tc_minus":[tc( get_cov_minus_i(self.cov,[i]),self.dim  ,self.use_det)
            for i in range(self.cov.shape[0] )  ] ,
               "dtc_minus": [dtc( get_cov_minus_i(self.cov,[i]),self.dim  ,self.use_det)
            for i in range(self.cov.shape[0] )  ] ,
               "g_tc":self.grad_tc() ,
               "g_dtc":self.grad_dtc(),
               "g_o_inf":self.grad_o_inf(),
               "g_s_inf":self.grad_s_inf(),
               "e_j": entropy(self.cov,self.dim,self.use_det)   , 
               "e_j_minus_i" :[entropy( get_cov_minus_i(self.cov,[i]),self.dim  ,self.use_det)
            for i in range(self.cov.shape[0] )  ] ,
               "e_minus_ji0" :[   entropy(get_cov_minus_i(self.cov,[0,i+1])  ,self.dim  ,self.use_det)
            for i in range(self.cov.shape[0] -1 )  ] ,        
               "e_j_i": [entropy( self.cov[i][i], self.dim  ,self.use_det)
            for i in range(self.cov.shape[0] )  ],
               "e_i_cond_slash": [
                   entropy(self.cov,self.dim,self.use_det) - entropy( get_cov_minus_i(self.cov,[i]),self.dim  ,self.use_det)
                   for i in range(self.cov.shape[0] )
               ]
               } 
 
    def tc(self):
        return  tc(self.cov,self.dim,self.use_det )

    def dtc(self):
        return  dtc(self.cov,self.dim ,self.use_det)

    def o_inf(self):
        return  o_inf(self.cov,self.dim,self.use_det)
    
    def s_inf(self):
        return  s_inf(self.cov,self.dim,self.use_det)
    
    def grad_tc(self):
        return  [tc(self.cov,self.dim,self.use_det ) - tc_minus for tc_minus in self.o_inf()[1] ]

    def grad_dtc(self):
        dtc_minus = [ dtc( get_cov_minus_i(self.cov,[i]),
                          self.dim,self.use_det)
            for i in range(self.cov.shape[0] )
        ] 
        return  [self.dtc() - dtc for dtc  in dtc_minus]

    def grad_o_inf(self):
        grad_tc = self.grad_tc()
        grad_dtc = self.grad_dtc()
        return [
            tc - dtc for tc,dtc in zip(grad_tc,grad_dtc)
        ]
    
    def grad_s_inf(self):
        grad_tc = self.grad_tc()
        grad_dtc = self.grad_dtc()
        return [
            tc + dtc for tc,dtc in zip(grad_tc,grad_dtc)
        ] 
        
    def o_inf_grad_n(self,n):
        o_info_grad = {}
        subsets = list(itertools.combinations(np.arange(self.nb_var), n))
      #  print(subsets)
        for sub in subsets:
            powerset = get_powerset(sub)
         #   print(powerset)
            o_grad = 0
            for index,set in enumerate(powerset):
                if len(set)>0:
                    cov_minus = get_cov_minus_i(self.cov,list(set) )
                else:
                    cov_minus = self.cov
                o_grad += (-1)**(len(set)) * o_inf(cov_minus,self.dim)
            o_info_grad[str(sub)] = o_grad
        return o_info_grad

   
    def sample_cov(self,N,dim,seed):
        ## extend the covariance matrix
        cov_d=eye(self.cov,dim)
        
        mu = np.zeros(cov_d.shape [0])
        samples = default_rng(seed=seed).multivariate_normal(mu, cov_d, size= (N) )
        samples = samples.reshape(N,self.nb_var,dim)

        #samples = np.transpose(samples, (0, 2, 1))
        # N x nb_var x dim  
        return  samples
    
    
    # def sample(self,N,dim,seed):
    #     mu = np.zeros(self.nb_var)
    #     samples = default_rng(seed=seed).multivariate_normal(mu, self.cov, size= (N,dim) )

    #     ## N*dim X nb_var 
      
    #    # N * dim * nb var 
    #   #  samples = samples.reshape(N,dim,self.nb_var)

    #     samples = np.transpose(samples, (0, 2, 1))
    #     # N x nb_var x dim  
    #     return  samples
    
    def get_torch_dataset(self,N,T,dim=1 ,rescale = False,seed = 42):
        
        print("starting to sample data")
        data = self.sample_cov(N+T,dim=dim,seed=seed)
        print("after cov")
        ##  N*dim X nb_var
        
        data = data.reshape(N+T, self.nb_var * dim)
        print("after reshape")
        if self.transformation =="H-C":
            print("applying transformation:"+ self.transformation)
            data = data * np.sqrt(np.abs( data)) 
        elif self.transformation =="CDF":
            print("applying transformation:"+ self.transformation)
            data = 0.5 * (1 + erf(data / 2**0.5))
        if rescale :
            ## (n_samples, n_features)
            data = preprocessing.scale(data)

        data = data.reshape(N+T, self.nb_var, dim)
       
        data_train = data[:N,:,:]
        data_test = data[N:,:,:] 
        
        return SynthetitcDataset(data_train),SynthetitcDataset(data_test)






class Task_redundant(Task):
    def __init__(self, nb_var=3, sigma=0.01,dim=1,use_det=False,normalized= True,rho = None):
        super().__init__(nb_var, sigma,dim,use_det)
        if normalized:
            self.build_cov_pure_redundancy_sigma_normalized(rho=rho)
            # self.build_cov_pure_redundancy_sigma()
    
    def build_cov_pure_redundancy_sigma_normalized(self,rho):   
        if rho==None:
            sigma = self.sigma 
            rho = 1/(1+sigma**2)
            
        cov = rho * np.ones((self.nb_var,self.nb_var))
        for i in range(self.nb_var):
            cov [i] [i] = 1
        self.cov = cov
        
        
        
    # def build_cov_pure_redundancy_sigma(self):   
    #     sigma = self.sigma 
    #     cov = np.zeros((self.nb_var,self.nb_var))

    #     for i in range(self.nb_var):
    #         if i ==0:
    #             cov[i][i] = sigma[0]
    #         else:
    #             cov[i][i] = sigma[0]**2 + sigma[i]**2
    #         for j in range(self.nb_var ):
    #             if i!=j:
    #                 cov[i][j] =  sigma[0]**2

    #     self.cov = cov

        



class Task_synergy(Task):
    def __init__(self, nb_var=3, sigma= 0.1,dim = 1,rho=None, normalized =True):
        super().__init__(nb_var, sigma, dim = dim)
        self.rho = rho

        if normalized:
            self.build_cov_pure_synergy_sigma_normalized()
    
    def build_cov_pure_synergy_sigma_normalized(self):   
        #sigma = self.sigma 
        
        nb_syn = 1 + self.nb_var-2

        rho = self.rho * 1/ np.sqrt(nb_syn) 
        
        cov = rho * np.zeros((self.nb_var,self.nb_var))
        
        cov [0][1]= 1/ np.sqrt(nb_syn)
        cov [1][0] = 1/ np.sqrt(nb_syn)
        
        for i in range(self.nb_var):
            cov [i][i] = 1
            if i>1:
                cov [0][i]=0
                cov [i][0]=0
                cov [1][i]=rho
                cov [i][1]=rho
        self.cov = cov

    # def build_cov_pure_synergy_sigma(self): 
    #     nb_var = self.nb_var  
    #     sigma = self.sigma
    #     sigma_s = self.sigma_s
    #     ## [sigma1 , sigma_2, sigma_s]  
    #     cov = np.zeros((self.nb_var,nb_var))

    #     cov[0][0] = cov[0][1] = cov[1][0] = sigma[0]**2
    #     cov[1][1] = sigma[0]**2 + np.sum([ s**2 for s in sigma_s] )

    #     for i in range(nb_var - 2 )  :
    #         cov[i+2][0] = cov[0][i+2]= 0
    #         cov[i+2][i+2] = sigma[i+2]**2 + sigma_s[i]**2 
    #     for i in range(nb_var - 2 ):
    #         cov[i+2][1] = cov[1][i+2]= sigma_s[i]**2 
    #     # if order >2:
    #     #     for i in range(nb_var - 4 )  :
    #     #         cov[i+4][i+3] =cov[i+3][i+4]= 0 
    #     #         cov[i+4][i+3] =cov[i+3][i+4]= 0
    #     self.cov = cov

    # # def sample(self,N,dim):
    #     order = len(self.sigma_s)
    #     X =[] 
    #     X_1 = np.random.normal(0,self.sigma[0] , (N,dim))
    #     X.append(X_1) 
    #     S = [np.random.normal(0,self.sigma_s[i] , (N,dim)) for i in range( len(self.sigma_s)) ] 
    #     X_2 = X_1 + np.sum(S,axis=0)
    #     X.append(X_2) 

    #     for i in range(len(self.sigma_s)):
    #         X.append( S[i]  +  np.random.normal(0,self.sigma[i+2] , (N,dim))   )

    #     samples = np.array(X)
    #     samples = np.transpose(samples, (1, 0, 2))
    #     return  samples
    
    # def build_cov_pure_synergy_order_1(self,order ):   
   
    #     # synergetic_info = [   for n in range( (self.nb_var // (synergy_elements) ) *synergy_elements ) ] 
    #     X =[]
    #     k =0
    #     for i in range(self.nb_var// (order+1 )):
    #         for i in range(order ) :
    #             X.append( np.random.normal(0,1, (N,)) )
    #         k+= order
    #         X_ = [ X[-(j+1)] for j in range(order)] 
    #         X.append( np.sum( X_,axis=0)  )  
    #         k+=1
                
    #     print(np.array(X).shape )
    #     self.cov = np.cov(X)
     

    


    # def build_cov_pure_synergy_order_2(self):   
    #     synergy_elements = 3
    #     synergetic_info = [ np.random.normal(0,1, (N,))  for n in range( (self.nb_var // (synergy_elements) ) * synergy_elements ) ] 
    #     ## (XN,XN-1,XN-2)-> XN-3 .. etc 

    #     X =[]
    #     X.append( synergetic_info[0] ) #  + self.noise_w * np.random.normal(0,1, (N,))  ) 

    #     for i in range(len(synergetic_info) - 2):
    #         X.append( synergetic_info[i] + (synergetic_info[i+1] + synergetic_info[i+2] )  )

    #         X.append( synergetic_info[i+1] + self.noise_w * np.random.normal(0,1, (N,)) )   
    #         X.append( synergetic_info[i+2] + self.noise_w * np.random.normal(0,1, (N,)) )  
        
    #     if len(X) != self.nb_var:
    #         self.nb_var = len(X)
    #         print("Nb variable not possible with number of interaction, nb_var changed to " + str(len(X) ))
    #     self.cov = np.cov(X)



class SynthetitcDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data):
        
        self.data = data
        self.nb_var = data.shape[1] 
    def __len__(self):
        return self.data.shape[0] 

    def __getitem__(self, idx):
        return { "x"+str(i) : torch.tensor( self.data[idx][i])  for i in range(self.nb_var)} 



class Task_combination(Task):
    def __init__(self, tasks = [], dim=1,use_det=True,normalized= True,transformation=None):
        cov = combined_cov( [task.cov for task in tasks])
        nb_var = cov.shape[0]
        self.tasks = tasks
        super().__init__(nb_var=nb_var, sigma=None,dim=dim,use_det=use_det,transformation=transformation)
        
        self.cov = cov
  

def fill_cov(new_cov,index,cov):
    for i in range(cov.shape[0]):
        for j in range(cov.shape[0]):
            new_cov [i+index] [j+index] = cov[i][j]
    return new_cov

def combined_cov(covs):
    total_dim = np.sum([c.shape [0] for c in covs])
    new_cov = np.zeros((total_dim,total_dim ))
    index = 0
    for i in range(len(covs) ) :
        new_cov = fill_cov(new_cov, index,covs [i]) 
        index+= covs[i].shape[0]
    return new_cov
        