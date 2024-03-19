import numpy as np
import torch
import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
#from src.model.score_net import UnetMLP
from src.libs.ema import EMA
from src.libs.SDE import VP_SDE ,concat_vect ,deconcat , get_normalizing_constant
from src.models.mlp import UnetMLP_simple
from src.libs.info_measures import get_grad_o_inf, get_tc,pop_elem_i ,get_joint_entropy  ,get_marg_entropy,get_dtc,infer_all_measures_2 
from src.models.transformer import DiT
from src.libs.util import *

T0 = 1







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
                 tx = True,
                 fill_zeros = False,
                 scores_order = 2,
                 batch_size=64,
                 margin_time = 1,
            
                 ):
        
        super(SOI, self).__init__()
        self.mod_list = mod_list
        self.gt = gt 
        self.tx = tx
        self.debug = debug
        self.weighted = weighted
        self.test_epoch = test_epoch
        self.margin_time = margin_time
        self.sizes = [mod_list[key] for key in mod_list.keys()]
        
        dim = np.sum(self.sizes)
               
        
        if dim <=10:
                hidden_dim = 128*2
               
        elif dim <=50:
                hidden_dim = 128*3
           
        elif dim <= 100:
                hidden_dim = 128*4
              
        else :
                hidden_dim = 128*5
        
        dim = np.max(self.sizes)    
        if dim <=5:
                htx = 12
        elif dim <= 10:
                htx = 18
        else :
                htx = 24
                
        time_dim = hidden_dim

       
        #self.score = UnetMLP(dim= dim , init_dim= hidden_dim ,dim_mults=[]  , time_dim= time_dim ,nb_mod= len(mod_list.keys()) )
        if self.tx:
            self.score = DiT(depth=4,
            hidden_size=htx * len(mod_list) ,
            mod_sizes= self.sizes, 
            mod_list= list(mod_list.keys()),
            num_heads=6,
            variable_input=False,
           # input_size=len(mod_list),
          #  in_channels=1
            )
        else:
            self.score = UnetMLP_simple(dim= dim , init_dim= hidden_dim ,dim_mults=[1,]  , time_dim= time_dim ,nb_mod= len(mod_list.keys()) )
      

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
                          nb_mod = len(self.mod_list),
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


    def score_inference(self,x,t=None,t_n=None,mask=None,std=None):
        with torch.no_grad():
            self.eval() 
            if self.use_ema:
                self.model_ema.module.eval()
                return self.model_ema.module(x,t=t,t_n=t_n,mask =mask,std=std)
            else:
                return self.score(x,t=t,t_n=t_n,mask =mask,std=std)

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
        if self.current_epoch % self.test_epoch == 0  and self.current_epoch  != 0:
             
           # r = self.compute_o_inf(data= self.test_samples, debias=False) 
            
            r = self.compute_o_inf_with_grad_mi(data= self.test_samples, debias=True,nb_iter= 1)
            r_uni = self.compute_o_inf_with_grad_mi(data= self.test_samples, debias=False,nb_iter= 1)
            
            print("O_inf -- GT: {}   , e : {}".format(self.gt["o_inf"], r["simple"]["o_inf"]) )
            print("Gradient O_inf -- GT: {}   , e : {}".format(self.gt["g_o_inf"], r["g_o_inf"]) )
            
            
            
            for met in ["tc","o_inf", "dtc"]:#  , "s_inf"]:  
                self.logger.experiment.add_scalars('Measures/{}'.format(met),  
                                                {'gt': self.gt[met] , 
                                                    'e': r["simple"][met] ,
                                                    'ue': r_uni["simple"][met] ,
                                                    }, global_step=self.global_step)
            if self.scores_order > 1 and self.debug:
                for index, mod in enumerate( self.mod_list) :
                    for met in ["g_o_inf"]:#["g_o_inf","tc_minus","dtc_minus"]:
                        self.logger.experiment.add_scalars('{}/i_{}'.format(met,mod) ,
                                                    {'gt': self.gt[met][index] , 
                                                        'e': r[met] [mod]["simple"] ,
                                                        'ue': r_uni[met] [mod]["simple"],
                                                        }, global_step=self.global_step)
                    
                    
                    # if j==0:
                    #     self.logger.experiment.add_scalars('debug_2/e_c_ji/i_0'.format(),  
                    #                                        r_grad["e_c_ij"]
                    #                             , global_step=self.global_step)
                
            if self.debug :
                for i in range(len(self.mod_list)):
         
                    
                    self.logger.experiment.add_scalars('Debbug/e_j_i{}'.format(i),  
                                                    {'gt': self.gt["e_j_i"] [i]  , 
                                                        'e': r["e_j_i"] [i],
                                                        'ue': r_uni["e_j_i"][i] , 
                                                    }, global_step=self.global_step)
                    
                    
                    self.logger.experiment.add_scalars('Debbug/e_j_cond_slash_{}'.format(i),  
                                                    {'gt': self.gt["e_j"] - self.gt["e_j_minus_i"] [i]  , 
                                                        'e': r["e_j_cond_slash"] [i],
                                                        'ue': r_uni["e_j_cond_slash"][i] , 
                                                    }, global_step=self.global_step)
                
                self.logger.experiment.add_scalars('Debbug/e_j',  
                                                    {'gt': self.gt["e_j"]  , 
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
        M = data["x0"].shape[0]  
        
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
            
         
        tc = { "simple": torch.stack([t["simple"] for t in tc_mc ]).mean() }
        
        dtc = { "simple": torch.stack([t["simple"] for t in dtc_mc ]).mean()
               }
        
        
        r = {
          "simple":  infer_all_measures_2(tc, dtc,type_= "simple")
        } 
        
        if self.debug:
            e_j = torch.stack(e_j_mc).mean().item()
            e_j_i = [ torch.stack(e_j_i_mc [key]).mean().item() for key in e_j_i_mc.keys() ]
            
            e_j_cond_slash = [ torch.stack(e_j_cond_slash_mc[key]).mean().item() for key in e_j_cond_slash_mc.keys() ]
            
            r ["e_j"] = e_j
            r ["e_j_i"] = e_j_i
            r ["e_j_cond_slash"] = e_j_cond_slash
            
            
        
        return r
        
        
        
        
    def compute_o_inf_with_grad(self,data, debias =False, eps = 1e-5, sigma =1.0, nb_iter = 10):
        
        self.sde.device = self.device
        self.score.eval()

        mods_list = list(data.keys())

        mods_sizes = [data[key].size(1) for key in mods_list ]
        data = { k :data[k].to(self.device) for k in mods_list}
        
        N = len(mods_list)
        M = data["x0"].shape[0]  
        
        e_j_mc = []
        
        tc_mc = []
        dtc_mc = []
        

        e_j_i_mc = {mod : [] for mod in data.keys()}
        e_j_cond_slash_mc = {mod : [] for mod in data.keys()}
        
        minus_tc_mc = {mod : [] for mod in data.keys()}
        minus_dtc_mc = {mod : [] for mod in data.keys()}
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

            marginals = {mod:marginalize_data(X_t, mod,fill_zeros=self.fill_zeros)  for mod in mods_list } 
            joint_marginals ={ mod:marginalize_one_mod(X_t, mod,fill_zeros=self.fill_zeros)  for mod in mods_list } 
            cond_x = { mod:cond_x_data(X_t,data, mod) for mod in mods_list } 
            
            cond_ij = { mod_i: { mod_j : marginalize_one_mod(cond_x[mod_i], mod_j, fill_zeros= self.fill_zeros ) 
                                 for mod_j in mods_list
                               } for mod_i in mods_list } 

            if debias:
                std_w = None
            else:
                std_w = std  

            with torch.no_grad():
                s_marg ={} 
                s_cond_x ={}  
                s_cond_ij = {}
                s_joint_marg = {}
                if self.tx:
                    
                    s_joint = - self.score_inference(x_t_joint.float(), t=t,
                                                     t_n= None,
                                                 mask = torch.ones_like(marg_masks[mods_list[0]]),
                                                 std= std_w).detach()
                    
                    for mod in mods_list:
            
                        s_marg[mod] =  - self.score_inference(marginals[mod],
                                                              t=t,
                                                              t_n= None,
                                                        mask =  marg_masks[mod] +
                                                        (-1) * (1 - marg_masks[mod] ) ,
                                                        std= std_w).detach()
                  
                        s_joint_marg [mod] = - self.score_inference(joint_marginals[mod], t=t,t_n= None, 
                                                         mask =(1- marg_masks[mod]) + (-1) * ( marg_masks[mod] ) ,
                                                         std=std_w).detach()
          
                        s_cond_x[mod] =  - self.score_inference(cond_x[mod], t=t,t_n= None,
                                                        mask = cond_x_mask[mod] , 
                                                        std=std_w).detach()
                        s_cond_ij [mod] = {}
                        for mod_j in mods_list:
                            if mod_j !=mod:
                                s_cond_ij [mod][mod_j] = - self.score_inference(cond_ij[mod][mod_j], t=t,t_n= None,
                                                         mask = cond_x_mask[mod] + (-1) *  marg_masks[mod_j]  , std= std_w).detach()
                            else:
                                s_cond_ij [mod][mod_j]= torch.zeros_like(s_cond_x[mod])
                else:
                    s_joint = - self.score_inference(x_t_joint.float(), 
                                                    t_n = t_n,std=std_w).detach()
                    for mod in mods_list:

                        s_marg[mod] =  - self.score_inference(concat_vect(marginals[mod]).float(), 
                                                            t_n=t_n * marg_masks[mod] + (self.margin_time) * (1 - marg_masks[mod] ) , std=std_w).detach()
                        
                        
                        s_joint_marg [mod] = - self.score_inference(concat_vect(joint_marginals[mod]).float(), 
                                                            t_n=t_n * (1- marg_masks[mod]) + (self.margin_time) * ( marg_masks[mod] ) , std=std_w).detach()
                        
                        s_cond_x[mod] =  - self.score_inference(concat_vect(cond_x[mod]).float(), 
                                                           t_n= t_n * cond_x_mask[mod] , std=std_w).detach()
                        s_cond_ij [mod] = {}
                        for mod_j in mods_list:
                            s_cond_ij [mod][mod_j] = - self.score_inference(concat_vect( cond_ij[mod][mod_j]).float(), 
                                                          t_n=  t_n * cond_x_mask[mod] + (self.margin_time) *  marg_masks[mod_j]  , std=std_w).detach()
                        
                        
            s_joint = deconcat(s_joint,mods_list,mods_sizes)

            s_marg ={
                mod : deconcat(s_marg[mod],mods_list,mods_sizes)[mod] 
                for mod in mods_list
            } 
        
            s_cond_x ={
                mod : deconcat(s_cond_x[mod],mods_list,mods_sizes)[mod]
                for mod in mods_list
            } 
            
            s_joint_marg = {
                mod : deconcat(s_joint_marg[mod],mods_list,mods_sizes)
                for mod in mods_list
            }
            
            s_cond_x_ij ={ 
                mod_j : { 
                    mod_i: deconcat(s_cond_ij[mod_i][mod_j],mods_list,mods_sizes)[mod_i]
                    for mod_i in mods_list 
                    } 
                for mod_j in mods_list
            } 

            if self.debug:
                e_j = get_joint_entropy(self.sde,s_joint=s_joint,
                                x_t = X_t, std=std, g= g, 
                                debias = debias,
                                x_0= data,
                                mean = mean, 
                                #sigma=sigma
                                )
                e_j_mc.append(e_j)
            
            tc = get_tc(self.sde,s_joint=s_joint, s_marg=s_marg, 
                                x_t = X_t, std=std, g= g, 
                                debias = debias,
                                mean = mean, 
                                #sigma=sigma
                                )   
            
            dtc = get_dtc(self.sde,s_joint=s_joint, 
                                s_cond=s_cond_x, 
                                x_t = X_t, std=std, g= g, 
                                debias = debias,
                                mean = mean, #sigma=sigma
                                )
            
            for grad_mod in mods_list:
      
                tc_i = get_tc(self.sde,s_joint=pop_elem_i( s_joint_marg[grad_mod],  [grad_mod]  ),
                                s_marg=pop_elem_i(  s_marg,[grad_mod] ), 
                                x_t = pop_elem_i(  X_t,[grad_mod] ), std=std, g= g, 
                                debias = debias,
                                mean = mean, #sigma=sigma
                                )
                minus_tc_mc[grad_mod].append(tc_i)
                dtc_i = get_dtc(self.sde,s_joint=pop_elem_i( s_joint_marg[grad_mod], [grad_mod]  ),
                                s_cond=pop_elem_i(  s_cond_x_ij[grad_mod],[grad_mod] ), 
                                x_t = pop_elem_i(  X_t,[grad_mod] ), std=std, g= g, 
                                debias = debias,
                                mean = mean, #sigma=sigma
                                )
                
                minus_dtc_mc[grad_mod].append( dtc_i )
                
            
                
                 
            
            
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
                                mean = mean, sigma=sigma
                                )
                    
                    e_j_cond_slash_mc[mod_minus_i].append(cond_e)
                    
                    marg_e =get_marg_entropy(self.sde,s_joint=s_marg [mod_minus_i],
                                x_t =  X_t[mod_minus_i] , 
                                x_0 = data[mod_minus_i] ,
                                std=std, g= g, 
                                debias = debias,
                                mean = mean, sigma=sigma)
                    
                    
                    e_j_i_mc[mod_minus_i].append(marg_e)
            
         
        tc = { "simple": torch.stack([t["simple"] for t in tc_mc ]).mean().item()}
        
        dtc = { "simple": torch.stack([t["simple"] for t in dtc_mc ]).mean().item()}
        
        tc_minus = { mod: { "simple":torch.stack( [ t["simple"] for t in minus_tc_mc[mod] ]).mean().item()   }  
                     for mod in mods_list }
        
        dtc_minus = { mod: { "simple":torch.stack( [ t["simple"] for t in minus_dtc_mc[mod] ]).mean().item()  }  
                     for mod in mods_list }
        
        g_o_inf = { mod: { "simple":  tc["simple"]-dtc["simple"]  - (  tc_minus[mod]["simple"] - dtc_minus[mod]["simple"] )  }  
                        for mod in mods_list } 
        r = {
          "simple":  infer_all_measures_2(tc, dtc,type_= "simple"),
          "tc_minus" :tc_minus,
          "dtc_minus": dtc_minus,
          "g_o_inf" :g_o_inf
        } 
        
        if self.debug:
            e_j = torch.stack(e_j_mc).mean().item()
            e_j_i = [ torch.stack(e_j_i_mc [key]).mean().item() for key in e_j_i_mc.keys() ]
            
            e_j_cond_slash = [ torch.stack(e_j_cond_slash_mc[key]).mean().item() for key in e_j_cond_slash_mc.keys() ]
            
            r ["e_j"] = e_j
            r ["e_j_i"] = e_j_i
            r ["e_j_cond_slash"] = e_j_cond_slash
        
        return r
        
        
        
        
        
        
    def compute_o_inf_with_grad_mi(self,data, debias =False, eps = 1e-3, sigma =1.0, nb_iter = 10):
        """Use simpler formulation to infer gradient o-information without Computing O-information but MI terms

        Args:
            data (_type_): _description_
            debias (bool, optional): _description_. Defaults to False.
            eps (_type_, optional): _description_. Defaults to 1e-3.
            sigma (float, optional): _description_. Defaults to 1.0.
            nb_iter (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        
        
        self.sde.device = self.device
        self.score.eval()

        mods_list = list(data.keys())

        mods_sizes = [data[key].size(1) for key in mods_list ]
        data = { k :data[k].to(self.device).float() for k in mods_list}
        
        N = len(mods_list)
        M = data["x0"].shape[0]  
        
        e_j_mc = []
        
        tc_mc = []
        dtc_mc = []
        

        e_j_i_mc = {mod : [] for mod in data.keys()}
        e_j_cond_slash_mc = {mod : [] for mod in data.keys()}
        
        g_o_inf = {mod : [] for mod in data.keys()}
        #minus_dtc_mc = {mod : [] for mod in data.keys()}
        for i in range(nb_iter):
            if debias:
                t = ( self.sde.sample_debiasing_t(shape=(M,1))  ).to(self.device)
            else:
                t = torch.round( ( (self.sde.T - eps) * torch.rand((M,1)) + eps ).to(self.device),decimals=3 )

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

            marginals = {mod:marginalize_data(X_t, mod,fill_zeros=self.fill_zeros)  for mod in mods_list } 
            joint_marginals ={ mod:marginalize_one_mod(X_t, mod,fill_zeros=self.fill_zeros)  for mod in mods_list } 
            cond_x = { mod:cond_x_data(X_t,data, mod) for mod in mods_list } 
            
            cond_ij = { mod_i: { mod_j : marginalize_one_mod(cond_x[mod_i], mod_j, fill_zeros= self.fill_zeros ) 
                                 for mod_j in mods_list
                               } for mod_i in mods_list } 

            if debias:
                std_w = None
            else:
                std_w = std  

            with torch.no_grad():
                s_marg ={} 
                s_cond_x ={}  
                s_cond_ij = {}
                s_joint_marg = {}
                if self.tx:
                    
                    s_joint = - self.score_inference(x_t_joint.float(), t=t,
                                                     t_n= None,
                                                 mask = torch.ones_like(marg_masks[mods_list[0]]),
                                                 std= std_w).detach()
                    
                    for mod in mods_list:
            
                        s_marg[mod] =  - self.score_inference(marginals[mod],
                                                              t=t,
                                                              t_n= None,
                                                        mask =  marg_masks[mod] +
                                                        (-1) * (1 - marg_masks[mod] ) ,
                                                        std= std_w).detach()
                  
                        s_joint_marg [mod] = - self.score_inference(joint_marginals[mod], t=t,t_n= None, 
                                                         mask =(1- marg_masks[mod]) + (-1) * ( marg_masks[mod] ) ,
                                                         std=std_w).detach()
          
                        s_cond_x[mod] =  - self.score_inference(cond_x[mod], t=t,t_n= None,
                                                        mask = cond_x_mask[mod] , 
                                                        std=std_w).detach()
                        s_cond_ij [mod] = {}
                        for mod_j in mods_list:
                            if mod_j !=mod:
                                s_cond_ij [mod][mod_j] = - self.score_inference(cond_ij[mod][mod_j], t=t,t_n= None,
                                                         mask = cond_x_mask[mod] + (-1) *  marg_masks[mod_j]  , std= std_w).detach()
                            else:
                                s_cond_ij [mod][mod_j]= torch.zeros_like(s_cond_x[mod])
                else:
                    s_joint = - self.score_inference(x_t_joint.float(), 
                                                    t_n = t_n,std=std_w).detach()
                    for mod in mods_list:

                        s_marg[mod] =  - self.score_inference(concat_vect(marginals[mod]).float(), 
                                                            t_n=t_n * marg_masks[mod] + (self.margin_time) * (1 - marg_masks[mod] ) , std=std_w).detach()
                        
                        
                        s_joint_marg [mod] = - self.score_inference(concat_vect(joint_marginals[mod]).float(), 
                                                            t_n=t_n * (1- marg_masks[mod]) + (self.margin_time) * ( marg_masks[mod] ) , std=std_w).detach()
                        
                        s_cond_x[mod] =  - self.score_inference(concat_vect(cond_x[mod]).float(), 
                                                           t_n= t_n * cond_x_mask[mod] , std=std_w).detach()
                        s_cond_ij [mod] = {}
                        for mod_j in mods_list:
                            s_cond_ij [mod][mod_j] = - self.score_inference(concat_vect( cond_ij[mod][mod_j]).float(), 
                                                          t_n=  t_n * cond_x_mask[mod] + (self.margin_time) *  marg_masks[mod_j]  , std=std_w).detach()
            
            
                                 
            s_joint = deconcat(s_joint,mods_list,mods_sizes)

            s_marg ={
                mod : deconcat(s_marg[mod],mods_list,mods_sizes)[mod] 
                for mod in mods_list
            } 
        
            s_cond_x ={
                mod : deconcat(s_cond_x[mod],mods_list,mods_sizes)[mod]
                for mod in mods_list
            } 
            
            s_joint_marg = {
                mod : deconcat(s_joint_marg[mod],mods_list,mods_sizes)
                for mod in mods_list
            }
            
            s_cond_x_ij ={ 
                mod_i : { 
                    mod_j: deconcat(s_cond_ij[mod_i][mod_j],mods_list,mods_sizes)[mod_i]
                    for mod_j in mods_list 
                    } 
                for mod_i in mods_list
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
                                mean = mean)   
            
            dtc = get_dtc(self.sde,s_joint=s_joint, 
                                s_cond=s_cond_x, 
                                x_t = X_t, std=std, g= g, 
                                debias = debias,
                                mean = mean)
            tc_mc.append(tc)
            dtc_mc.append(dtc)
            
            for grad_mod in mods_list:
                # g_tc =  get_tc_minus(self.sde,s_joint=s_joint, 
                #                 s_marg=s_marg, 
                #                 s_cond = s_cond_x, 
                #                 grad_mod = grad_mod, 
                #                 x_t = X_t, std=std, g= g, 
                #                 debias = debias,
                #                 mean = mean, sigma=sigma) 
                # minus_tc_mc[grad_mod].append(g_tc)
                
                g_o = get_grad_o_inf(
                                self.sde,
                                s_marg=s_marg, 
                                s_con_i = s_cond_x,
                                s_cond_ij = s_cond_x_ij,
                                grad_mod=grad_mod,
                                x_t = X_t, 
                                std=std, g= g, 
                                debias = debias,
                                mean = mean)
                g_o_inf[grad_mod].append(g_o)  
               
               
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
            
         
        tc = { "simple": torch.stack([t["simple"] for t in tc_mc ]).mean().item()}
        
        dtc = { "simple": torch.stack([t["simple"] for t in dtc_mc ]).mean().item()}
        
        g_o = { mod: {
               "simple":   torch.stack([t["simple"] for t in g_o_inf[mod] ]).mean().item()  
              }  for mod in g_o_inf.keys()
               }
            
        
        r = {
          "simple":  infer_all_measures_2(tc, dtc,type_= "simple"),
          
        #   "tc_minus" :tc_minus,
        #   "dtc_minus": dtc_minus,
          "g_o_inf" :g_o
        } 
        
        if self.debug:
            e_j = torch.stack(e_j_mc).mean().item()
            e_j_i = [ torch.stack(e_j_i_mc [key]).mean().item() for key in e_j_i_mc.keys() ]
            
            e_j_cond_slash = [ torch.stack(e_j_cond_slash_mc[key]).mean().item() for key in e_j_cond_slash_mc.keys() ]
            
            r ["e_j"] = e_j
            r ["e_j_i"] = e_j_i
            r ["e_j_cond_slash"] = e_j_cond_slash
        
        return r
        
        
        
        
        
        

    
        
        
        
    def compute_o_inf_batch(self,test_loader, debias =False, eps = 1e-5, sigma =1.0, nb_iter = 10):

        mets = ["o_inf","s_inf","tc","dtc"]
        out = {
            "simple": {met :[] for met in mets},
            "sigma":{met :[] for met in mets}
        }
       
        for batch in tqdm(test_loader):
            r = self.compute_o_inf(batch,debias=debias,nb_iter=nb_iter)
            for met in mets:
               out["simple"] [met].append(r["simple"][met]) 
               out["sigma"] [met].append(r["sigma"][met]) 
        
        for met in mets:
            
            out["simple"] [met] =  torch.stack( out["simple"] [met] ).mean().cpu().numpy().item()
            out["sigma"] [met] =  torch.stack( out["sigma"] [met] ).mean().cpu().numpy().item()
        return out






