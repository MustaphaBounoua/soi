import numpy as np
import torch
import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
#from src.model.score_net import UnetMLP
from src.libs.ema import EMA
from libs.SDE import VP_SDE ,concat_vect ,deconcat , get_normalizing_constant
from models.mlp import UnetMLP_simple
from src.libs.info_measures import minus_x_data, get_tc,pop_elem_i ,get_joint_entropy  ,get_marg_entropy,get_dtc,infer_all_measures_2 
learning_rate = 1e-4



T0 = 1
vtype = 'rademacher'
lr = 0.001



def marginalize_data(x_t, mod,fill_zeros =False):
    x = x_t.copy()
    for k in x.keys():
        if k !=mod:
            if fill_zeros:
                x[k]=torch.zeros_like(x_t[k] ) 
            else:
                x[k]=torch.randn_like(x_t[k] )
    return x

def cond_x_data(x_t,data,mod):
    x = x_t.copy()
    for k in x.keys():
        if k !=mod:
            x[k]=data[k] 
    return x


class SOI(pl.LightningModule):
    
    def __init__(self,
                 lr = 1e-3,
                 mod_list={ "x":1,"y":1,"m":1} ,
                 debias = False, 
                 weighted = False,
                 use_ema = False ,
                 weight_subsets = True,
                 test_samples = None,
                 test_epoch =25,
                 gt = None,
                 debug =False,
                 fill_zeros = False,
                 scores_order = 0,
                 batch_size=64,
                 margin_time = 1,
                 use_only_margin =False,
            
                 ):
        
        super(SOI, self).__init__()
        self.mod_list = mod_list
        self.gt = gt 
        self.debug = debug
        self.weighted = weighted
        self.test_epoch = test_epoch
        self.margin_time = margin_time
        dim = 0
        for key in mod_list.keys():
                dim+=mod_list[key] 
            
        
        if dim <=30:
                hidden_dim = 128
        elif dim <=75:
                hidden_dim = 192
        elif dim <= 150:
                hidden_dim = 256
        else :
                hidden_dim = 128*3

        time_dim = hidden_dim

       
        #self.score = UnetMLP(dim= dim , init_dim= hidden_dim ,dim_mults=[]  , time_dim= time_dim ,nb_mod= len(mod_list.keys()) )
        
        self.score = UnetMLP_simple(dim= dim , init_dim= hidden_dim ,dim_mults=[]  , time_dim= time_dim ,nb_mod= len(mod_list.keys()) )
      

        #self.score = UnetMLP_s(dim= dim , init_dim= hidden_dim, bottelneck= hidden_dim, time_dim= time_dim ,nb_mod= len(mod_list.keys()) )
      
        self.debias = debias
        self.lr = lr
        self.weight_subsets =weight_subsets
        self.use_ema = use_ema
        self.gt = gt
        self.fill_zeros =fill_zeros
        self.scores_order = scores_order
        self.save_hyperparameters("weight_subsets","debias","lr","use_ema","batch_size","mod_list")

        self.test_samples =  test_samples
        
        self.model_ema = EMA(self.score, decay=0.999) if use_ema else None
        self.sde = VP_SDE(importance_sampling = self.debias ,
                           fill_zeros= fill_zeros,
                          liklihood_weighting = False,
                          nb_mod = len(self.mod_list),
                          use_only_margin = use_only_margin,
                          weight_subsets= weight_subsets,
                          scores_order = scores_order,
                          margin_time = margin_time)
        
        
    def training_step(self, batch, batch_idx):
       
        self.train()
     
        loss = self.sde.train_step(batch,self.score).mean()  # forward and compute loss

        self.log("loss",loss)

        return {"loss":loss}
    


    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update(self.score)


    def score_inference(self,x,t,std):
        with torch.no_grad():
            self.eval() 
            if self.use_ema:
                self.model_ema.module.eval()
                return self.model_ema.module(x,t,std)
            else:
                return self.score(x,t,std)

    def validation_step(self, batch, batch_idx):
        self.eval()
 
        loss = self.sde.train_step(batch,self.score).mean()  # # forward and compute loss
        self.log("loss_test",loss)
        return {"loss":loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.score.parameters(), lr= self.lr  )
        return optimizer

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if self.current_epoch % self.test_epoch == 0 :
             
           # r = self.compute_o_inf(data= self.test_samples, debias=False) 
            print(self.debug )
            if self.debug ==False:
                print("okay")
                r = self.compute_o_inf_batch(test_loader= self.test_samples, debias=True,nb_iter= 1)
                r_uni = self.compute_o_inf_batch(test_loader= self.test_samples, debias=False,nb_iter= 1)
            else:
                r = self.compute_o_inf(data= self.test_samples, debias=True,nb_iter= 10)
                r_uni = self.compute_o_inf(data= self.test_samples, debias=False,nb_iter= 10)
                
            
            for met in ["tc","o_inf", "dtc"]:#  , "s_inf"]:  
                self.logger.experiment.add_scalars('Measures/{}'.format(met),  
                                                {'e': r["simple"][met] ,
                                                    'ue': r_uni["simple"][met] ,
                                                    }, global_step=self.global_step)
            if self.scores_order>1 and self.debug:
                for j in  range(len(self.mod_list)):
                    r_grad = self.compute_grad_1_o_inf(self.test_samples,grad_i=j,debias=True)
                    r_grad_u = self.compute_grad_1_o_inf(self.test_samples,grad_i=j,debias=True)
                    for met in ["g_tc","g_o_inf", "g_dtc","g_s_inf"]:
                        self.logger.experiment.add_scalars('Measures/Gradient/i_/{}'.format(j,met),  
                                                { 'e': r_grad["simple"][met] ,
                                                    'ue': r_grad_u["simple"][met] ,
                                                    }, global_step=self.global_step)
                
            if self.debug :
                for i in range(len(self.mod_list)):
         
                    
                    self.logger.experiment.add_scalars('Debbug/e_j_i{}'.format(i),  
                                                    {
                                                        'e': r["e_j_i"] [i],
                                                        'ue': r_uni["e_j_i"][i] , 
                                                    }, global_step=self.global_step)
                    
                    
                    self.logger.experiment.add_scalars('Debbug/e_j_cond_slash_{}'.format(i),  
                                                    {
                                                        'e': r["e_j_cond_slash"] [i],
                                                        'ue': r_uni["e_j_cond_slash"][i] , 
                                                    }, global_step=self.global_step)
                
                self.logger.experiment.add_scalars('Debbug/e_j',  
                                                    {
                                                        'e': r["e_j"] ,
                                                        'ue': r_uni["e_j"] , 
                                                    }, global_step=self.global_step)
                
               
            
    
    def compute_o_inf(self,data, debias =False, eps = 1e-5, sigma =1.0, nb_iter = 10):
        
        self.sde.device = self.device
        self.score.eval()

        mods_list = list(data.keys())

        mods_sizes = [data[key].size(1) for key in mods_list ]
        data = { k :data[k].to(self.device) for k in mods_list}
        
        N = len(mods_list)
        M = data[mods_list [0]].shape[0]  
        
        e_j_mc = []
        
        tc_mc = []
        dtc_mc = []
        

        e_j_i_mc = {mod : [] for mod in data.keys()}
        e_j_cond_slash_mc = {mod : [] for mod in data.keys()}
        for i in range(nb_iter):
            if debias:
                t = ( self.sde.sample_debiasing_t(shape=(M,1))  ).to(self.device)
            else:
                t = ( (self.sde.T - eps) * torch.rand((M,1)) + eps ).to(self.device)

            t_n = t.expand((M,N ) )
            X_t,z_m, mean, std ,f, g = self.sde.sample(t_n,t, data, mods_list)

            
            marg_masks = {}
            cond_x_mask = {} 

            for i,mod in enumerate(mods_list): 
                mask =[0] * N 
                mask[i] = 1  
                marg_masks[ mod] = torch.tensor( mask ).to(self.device).expand(t_n.size())

                mask =[0] * N 
                mask[i] = 1 
                cond_x_mask[ mod] = torch.tensor( mask ).to(self.device).expand(t_n.size())

            
    
            x_t_joint = concat_vect(X_t)

            marginals = {mod:concat_vect(marginalize_data(X_t, mod,fill_zeros=self.fill_zeros) ) for mod in mods_list } 
        
            cond_x = { mod:concat_vect(cond_x_data(X_t,data, mod)) for mod in mods_list } 

            if debias:
                std_w = None
            else:
                std_w = std  

            with torch.no_grad():

                s_joint = - self.score_inference(x_t_joint.float(), t_n, std_w).detach()
                s_marg ={} 
                s_cond_x ={}  
                for mod in mods_list:

                    s_marg[mod] =  - self.score_inference(marginals[mod].float(), 
                                                        t_n * marg_masks[mod] + (self.margin_time) * (1 - marg_masks[mod] ) , std_w).detach()
                    
                    s_cond_x[mod] =  - self.score_inference(cond_x[mod].float(), 
                                                        t_n * cond_x_mask[mod] , std_w).detach()
                        
            s_joint = deconcat(s_joint,mods_list,mods_sizes)

            s_marg ={
                mod : deconcat(s_marg[mod],mods_list,mods_sizes)[mod] 
                for mod in mods_list
            } 
        
            s_cond_x ={
                mod : deconcat(s_cond_x[mod],mods_list,mods_sizes)[mod]
                for mod in mods_list
            } 

            if self.debug:
                e_j = get_joint_entropy(self.sde,s_joint=s_joint,
                                x_t = X_t, std=std, g= g, 
                                debias = debias,
                                x_0= data,
                                mean = mean, 
                                sigma=sigma)
                e_j_mc.append(e_j)
            
            tc = get_tc(self.sde,s_joint=s_joint, s_marg=s_marg, 
                                x_t = X_t, std=std, g= g, 
                                debias = debias,
                                mean = mean, sigma=sigma)   
            
            dtc = get_dtc(self.sde,s_joint=s_joint, s_cond=s_cond_x, 
                                x_t = X_t, std=std, g= g, 
                                debias = debias,
                                mean = mean, sigma=sigma)
            tc_mc.append(tc)
            dtc_mc.append(dtc)
            

            # e_j_i = []
            # e_j_cond_slash = []
            if self.debug:
                
                for index, mod_minus_i in enumerate(mods_list):
                    cond_e = get_marg_entropy(self.sde,s_joint=s_cond_x[mod_minus_i],
                                x_t =  X_t[mod_minus_i] , 
                                x_0 = data[mod_minus_i] ,
                                std=std, g= g, 
                                debias = debias,
                                mean = mean, sigma=sigma)
                    
                    e_j_cond_slash_mc[mod_minus_i].append(cond_e)
                    
                    marg_e =get_marg_entropy(self.sde,s_joint=s_marg [mod_minus_i],
                                x_t =  X_t[mod_minus_i] , 
                                x_0 = data[mod_minus_i] ,
                                std=std, g= g, 
                                debias = debias,
                                mean = mean, sigma=sigma)
                    
                    
                    e_j_i_mc[mod_minus_i].append(marg_e)
            
         
        tc = { "simple": torch.stack([t["simple"] for t in tc_mc ]).mean(),
               }
        
        dtc = { "simple": torch.stack([t["simple"] for t in dtc_mc ]).mean(),
                }
        
        
        r = {
          "simple":  infer_all_measures_2(tc, dtc,type_= "simple"),
          
        } 
        
        if self.debug:
            e_j = torch.stack(e_j_mc).mean().item()
            e_j_i = [ torch.stack(e_j_i_mc [key]).mean().item() for key in e_j_i_mc.keys() ]
            
            e_j_cond_slash = [ torch.stack(e_j_cond_slash_mc[key]).mean().item() for key in e_j_cond_slash_mc.keys() ]
            
            r ["e_j"] = e_j
            r ["e_j_i"] = e_j_i
            r ["e_j_cond_slash"] = e_j_cond_slash
        return r
        

    def compute_grad_1_o_inf(self,data,grad_i, debias =False, eps = 1e-5, sigma =1.0, nb_iter = 10):
        
        self.sde.device = self.device
        self.score.eval()
        
        mods_list = list(data.keys())

        mods_sizes = [data[key].size(1) for key in mods_list ]
        data = { k :data[k].to(self.device) for k in mods_list}
        grad_key = mods_list [grad_i]
        N = len(mods_list)
        M = data["x0"].shape[0]  
        
        mask =[0] * N 
        mask[grad_i] = 1  
        marg_mask = torch.tensor( mask ).to(self.device).expand((M,N))
            
        cond_x_mask_i = {}
            
            
        for i,mod in enumerate(mods_list): 
            mask =[0] * N 
            mask[i] = 1 
            cond_x_mask_i [mod]= mask
        
        grad_tc_mc = []
        grad_dtc_mc = []
        

        for i in range(nb_iter):
            if debias:
                t = ( self.sde.sample_debiasing_t(shape=(M,1))  ).to(self.device)
            else:
                t = ( (self.sde.T - eps) * torch.rand((M,1)) + eps ).to(self.device)

            t_n = t.expand((M,N ) )
            X_t,z_m, mean, std ,f, g = self.sde.sample(t_n,t, data, mods_list)

            marginal = marginalize_data(X_t, grad_key,fill_zeros=self.fill_zeros)
            
            cond_x  = {mod:  concat_vect( cond_x_data(X_t,data, mod ) )  for mod in mods_list }
            
            cond_ij  = {mod: concat_vect( marginalize_data( cond_x_data(X_t,data, mod ),
                                              mod = grad_key, fill_zeros= self.fill_zeros ) )
                        for mod in mods_list if mod!=grad_key}
            
            if debias:
                std_w = None
            else:
                std_w = std  

            with torch.no_grad():

                s_marg = - self.score_inference( concat_vect(marginal).float(), t_n * marg_mask + self.margin_time * (1-marg_mask), std_w).detach()
                 
                s_cond_x_i ={}  
                s_cond_x_ij ={}
                for index, mod in enumerate( mods_list ):
                    print(cond_x_mask_i)
                    s_cond_x_i[mod] =  - self.score_inference( cond_x[mod].float(), 
                                                        t_n * cond_x_mask_i[mod] , std_w).detach()
                    
                    if index != grad_i:
                        s_cond_x_ij[mod] =  - self.score_inference(cond_ij[mod].float(), 
                                                        t_n * cond_x_mask_i[mod] + self.margin_time * marg_mask, std_w).detach()
                        
            s_marg = deconcat(s_marg,mods_list,mods_sizes)[grad_key]

            s_cond_x_i ={
                mod : deconcat(s_cond_x_i[mod],mods_list,mods_sizes)[mod] 
                for mod in mods_list
            } 
        
            s_cond_x_i_j ={
                mod : deconcat(s_cond_x_ij[mod],mods_list,mods_sizes)[mod] 
                for mod in s_cond_x_ij.keys()
            } 
            
            
            grad_tc = get_grad_tc(self.sde,s_marg=s_marg, 
                                  s_cond=s_cond_x_i[grad_key], 
                                x_t = X_t, std=std, g= g, 
                                debias = debias,
                                mean = mean, sigma=sigma)   
            
            grad_dtc = get_grad_dtc(self.sde,s_cond_i= pop_elem_i(s_cond_x_i,[grad_key] ) , 
                                    s_cond_ij=s_cond_x_i_j, 
                                x_t = pop_elem_i(X_t,[grad_key]), 
                                std=std, g= g, 
                                debias = debias,
                                mean = mean, sigma=sigma)
            
            grad_tc_mc.append(grad_tc)
            grad_dtc_mc.append(grad_dtc)
    
            
         
        g_tc = { "simple": torch.stack([t["simple"] for t in grad_tc ]).mean(),
              }
        
        g_dtc = { "simple": torch.stack([t["simple"] for t in grad_dtc ]).mean(),
               }
        
        
        r = {
          "simple":  infer_all_measures_grad(g_tc, g_dtc,type_= "simple"),
         
        } 
        
        return r

        
        
        
    def compute_o_inf_batch(self,test_loader, debias =False, eps = 1e-5, sigma =1.0, nb_iter = 10):

        mets = ["o_inf","s_inf","tc","dtc"]
        out = {
            "simple": {met :[] for met in mets},
     
        }
       
        for batch in tqdm(test_loader):
            r = self.compute_o_inf(batch,debias=debias,nb_iter=nb_iter)
            for met in mets:
               out["simple"] [met].append(r["simple"][met]) 
        
        
        for met in mets:
            
            out["simple"] [met] =  torch.stack( out["simple"] [met] ).mean().cpu().numpy().item()
   
        return out






