import torch


def deconcat(z,mod_list,sizes):
    z_mods={}
    idx=0
    for i,mod in enumerate( mod_list):
        z_mods[mod] = z[:,idx:idx+ sizes[i] ]
        idx +=sizes[i]
    return z_mods


def concat_vect(encodings):
    z = torch.Tensor()
    for key in encodings.keys():
        z = z.to(encodings[key].device)
        z = torch.cat( [z, encodings[key]],dim = -1 )
    return z 



def unsequeeze_dict(data):
        for key in data.keys():
            if data[key].ndim == 1 :
                data[key]= data[key].view(data[key].size(0),1)
        return data

def marginalize_one_mod(x_t, mod,fill_zeros =False):
    x = x_t.copy()
    for k in x.keys():
        if k ==mod:
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








def marginalize_data(x_t, mod,fill_zeros =False):
    x = x_t.copy()
    for k in x.keys():
        if k !=mod:
            if fill_zeros:
                x[k]=torch.zeros_like(x_t[k] ) 
            else:
                x[k]=torch.randn_like(x_t[k] )
    return x