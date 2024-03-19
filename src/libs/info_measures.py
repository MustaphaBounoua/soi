import numpy as np
import torch
from src.libs.util import concat_vect, deconcat  
from src.libs.importance import get_normalizing_constant



def minus_x_data(x_t, mod,fill_zeros=True):
        x = x_t.copy()
        for k in x.keys():
                if k ==mod:
                    if fill_zeros:
                        x[k]=torch.zeros_like(x_t[k] ) 
                    else:
                        x[k]=torch.rand_like(x_t[k] )
        return x



def pop_elem_i(encodings , i =[]  ):
    encodings = encodings.copy()
    return{
        key : encodings[key] for key in encodings.keys() if ( key in i ) == False
    } 

def get_marg_entropy(sde, s_joint , x_t,x_0,  std, mean, g ,debias,sigma =1.0 ):
    khi_t =   mean **2 * sigma **2 + std**2 
    joint_0 =x_0 

    M = joint_0.shape[0]
    N = joint_0.shape[1]  
    
    term = N*0.5*np.log(2 *np.pi ) + 0.5* torch.sum(joint_0**2)/M # - 0.5 * N * torch.sum( torch.log(khi_t) -1 +  1 / khi_t ) 
    if debias:

        const = get_normalizing_constant((1,)).to(sde.device)
        h_joint = term - const *0.5* ((s_joint + std * x_t  /  khi_t  )**2).sum()/ M   
    else:
        h_joint = term - 0.5* (g**2*(s_joint + x_t  /  khi_t  )**2).sum()/ M  
    return h_joint


def get_joint_entropy(sde, s_joint , x_t,x_0,  std, mean, g ,debias,sigma =1.0 ):
    khi_t = - (  mean **2 * sigma **2 + std**2 )
    joint_0 =concat_vect(x_0) 

    M = joint_0.shape[0]
    N = joint_0.shape[1]  
    
    term = N*0.5*np.log(2 *np.pi ) + 0.5* torch.sum(joint_0**2)/M # - 0.5 * N * torch.sum( torch.log(khi_t) -1 +  1 / khi_t ) 
    if debias:

        const = get_normalizing_constant((1,)).to(sde.device)
        
        h_joint = term - const *0.5* ((concat_vect(s_joint) - std * concat_vect(x_t)  /  khi_t  )**2).sum()/ M   

    else:
       
        h_joint = term - 0.5* (g**2*(concat_vect(s_joint) - concat_vect(x_t)  /  khi_t  )**2).sum()/ M  

    return h_joint



def get_tc(sde, s_joint ,s_marg, x_t,  std, mean, g ,debias):

    M = std.shape[0] 

    if debias:

        const = get_normalizing_constant((1,)).to(sde.device)
        tc = const *0.5* ((concat_vect(s_joint) - concat_vect(s_marg)  )**2).sum()/ M
    else:
        tc = 0.5* (g**2*(concat_vect(s_joint) - concat_vect(s_marg)  )**2).sum()/ M
    return {"simple" :tc   }



def get_dtc(sde, s_joint ,s_cond, x_t,  std, mean, g ,debias):

    M = std.shape[0] 

    if debias:
        const = get_normalizing_constant((1,)).to(sde.device)
        dtc = const *0.5* ((concat_vect(s_joint) - concat_vect(s_cond)  )**2).sum()/ M
    else:
       
        dtc = 0.5* (g**2*(concat_vect(s_joint) - concat_vect(s_cond)  )**2).sum()/ M

    return {"simple" :dtc    }


    
    
    
    
def get_grad_o_inf(sde, s_marg ,s_con_i ,s_cond_ij, x_t,grad_mod,  std, mean, g ,debias,sigma =1.0 ):
    khi_t =   mean **2 * sigma **2 + std**2 
    
    M = std.shape[0] 

    if debias:

        const = get_normalizing_constant((1,)).to(sde.device)
        
        # h_x_i_marg =  -const *0.5* ((s_marg [grad_mod]  + std * x_t[grad_mod]  /  khi_t  )**2).sum()/ M 
        # h_x_i_cond =  -const *0.5* ((s_con_i[grad_mod]  + std * x_t[grad_mod]  /  khi_t  )**2).sum()/ M
        
        # i_x_i =  h_x_i_marg  - h_x_i_cond
        
        
        # # h_x_minus_i_marg =  const *0.5* ( (concat_vect( pop_elem_i(s_marg,[grad_mod]) ) + 
        # #                                               std * concat_vect(pop_elem_i(x_t,[grad_mod] ) ) /  khi_t  )
        # #                                  **2).sum()/ M
        
        # h_x_minus_ii_cond =  - torch.stack([const *0.5* (( s_cond_ij[grad_mod][mod_j] + 
        #                                     std * x_t[grad_mod]  /  khi_t  )**2).sum()/ M for mod_j 
        #                                     in s_marg.keys() if mod_j != grad_mod ] ).sum()
        
        
        # i_xij =  (len(s_marg.keys())-1) * h_x_i_marg -  h_x_minus_ii_cond
        
        # o_inf_sigma =  (2-len(s_marg.keys())  ) * i_x_i + i_xij


        i_x_i = const *0.5* (( s_marg [grad_mod] - s_con_i [grad_mod]  )**2).sum()/ M
        
        i_xij =  torch.stack([const *0.5* (( s_marg[grad_mod] -s_cond_ij[grad_mod][mod_j] )**2).sum()/ M
                                for mod_j in s_marg.keys() if mod_j != grad_mod ]).sum()
        
        #i_xij = const *0.5* ((concat_vect(pop_elem_i(s_marg,[grad_mod]) ) - concat_vect( pop_elem_i(s_cond_ij[grad_mod],[grad_mod]) ) )**2).sum()/ M
        o_inf=  (2-len(s_marg.keys())  ) * i_x_i + i_xij
       
    else:
        
        # h_x_i_marg =  0.5* (g**2*(s_marg [grad_mod]  +  x_t[grad_mod]  /  khi_t  )**2).sum()/ M 
        # h_x_i_cond =  0.5* (g**2*(s_con_i [grad_mod]  +  x_t[grad_mod]  /  khi_t  )**2).sum()/ M
        # i_x_i = - h_x_i_marg  + h_x_i_cond
        
        
        # h_x_minus_i_marg =  0.5* (g**2*(concat_vect( pop_elem_i(s_marg,[grad_mod]) ) +  concat_vect(pop_elem_i(x_t,[grad_mod] ) )  /  khi_t  )**2).sum()/ M
        # h_x_minus_ii_cond =  0.5* (g**2*(concat_vect( pop_elem_i(s_cond_ij[grad_mod],[grad_mod]) ) + concat_vect(pop_elem_i(x_t,[grad_mod] ) ) /  khi_t  )**2).sum()/ M
        # i_xij = - h_x_minus_i_marg + h_x_minus_ii_cond
        # o_inf_sigma =  (2-len(s_marg.keys())  ) * i_x_i + i_xij

        i_x_i = 0.5* (g**2*( s_marg [grad_mod] - s_con_i [grad_mod]  )**2).sum()/ M
        i_xij = 0.5* (g**2*(concat_vect( pop_elem_i(s_marg,[grad_mod])) - concat_vect( pop_elem_i(s_cond_ij[grad_mod],[grad_mod]) ) )**2).sum()/ M
        o_inf=  (2-len(s_marg.keys())  ) * i_x_i + i_xij
        

    return {"simple" :o_inf   }
    
    
    
    
def infer_all_measures_2(tc,dtc, type_ = "simple"):

    return {
                "tc": tc[type_],
                "dtc":dtc[type_],
                "o_inf": tc[type_] - dtc[type_] ,
                "s_inf" :tc[type_] + dtc[type_],
            #    "tc_minus": tc_minus_i
             } 
    
       

    