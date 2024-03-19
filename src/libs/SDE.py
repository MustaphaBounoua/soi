
import torch
import itertools
import numpy as np
from .util import *
from .importance import *


class VP_SDE():
    def __init__(self,
                 beta_min=0.1,
                 beta_max=20,
                 N=1000,
                 importance_sampling=True,
                 nb_mod=2,
                 scores_order=0,
                 fill_zeros=False,
                 weight_subsets=True,
                 margin_time=1,
                 ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.margin_time = margin_time
        self.N = N
        self.fill_zeros = fill_zeros
        self.T = 1
        self.importance_sampling = importance_sampling
        self.device = "cuda"
        self.nb_mod = nb_mod
        self.t_epsilon = 1e-3
        self.weight_subsets = weight_subsets

        self.subsets = self.get_subsets_joint_marginal_cond(
            scores_order=scores_order)

    def get_subsets_joint_marginal_cond(self, scores_order):
        nb_mod = self.nb_mod
        subsets = [list(i)
                   for i in itertools.product([0, 1, -1], repeat=nb_mod)]
        if scores_order == 1:
            subsets = [s for s in subsets if (sum(s) == nb_mod) or  # joint
                       (np.sum(np.array(s) == 1) and np.sum(np.array(s) == -1)
                        == nb_mod-1)  # marginal with cardinal =1
                       or (np.sum(np.array(s) == 1) == 1 and np.min(np.array(s)) == 0)]  # cond
        elif scores_order == 2:
            subsets =[ s for s in subsets if  
                      ( sum(s) == nb_mod )  or  ##joint
                    ( np.sum(np.array(s) == 1) == nb_mod-1 and np.sum( np.array(s) == -1) == 1 )  ##marginal 
                    or 
                    ( np.sum( np.array(s) == 1) == 1 and np.sum(np.array(s)==0 ) == nb_mod-1  )  #cond full
                    or ( np.sum( np.array(s) == 1) == 1 and np.sum(np.array(s)==0 ) == nb_mod-2  )   #cond_ij    
                    or   ( np.sum( np.array(s) == 1) == 1 and np.sum(np.array(s)==-1 ) == nb_mod-1  ) # marginal
                    ]   

        if self.weight_subsets:
            subsets_w = []
            if scores_order == 1:
                print("Weighting the scores to learn ")
                for s in subsets:
                    nb_var_inset = np.sum(
                        np.array(s) == 1) + np.sum(np.array(s) == 0)//2
                    for i in range(nb_var_inset):
                        subsets_w.append(s)
                subsets = subsets_w
            else:
                for s in subsets:
                    if np.sum(np.array(s) == 1) == self.nb_mod:
                        nb_var_inset = 4
                    elif (np.sum(np.array(s) == 1) == (nb_mod-1) and np.sum(np.array(s) == -1) == 1):
                        nb_var_inset = 3
                    elif (np.sum(np.array(s) == 0)) == self.nb_mod-1:
                        nb_var_inset = 2
                    elif (np.sum(np.array(s) == 0)) == self.nb_mod-2:
                        nb_var_inset = 2
                    elif (np.sum(np.array(s) == 0)) == 0:
                        nb_var_inset = 1
                    # nb_var_inset = 1
                for i in range(nb_var_inset):
                    subsets_w.append(s)
            subsets = subsets_w
        np.random.shuffle(subsets)
        return torch.tensor(subsets).to(self.device)

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, t):
        return -0.5*self.beta_t(t), torch.sqrt(self.beta_t(t))

    def marg_prob(self, t, x):
        # return mean std of p(x(t))
        log_mean_coeff = -0.25 * t ** 2 * \
            (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

        log_mean_coeff = log_mean_coeff.to(self.device)

        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        return mean.view(-1, 1) * torch.ones_like(x).to(self.device), std.view(-1, 1) * torch.ones_like(x).to(self.device)

    def sample(self, t, time, data, mods_list):

        x_t_m = {}
        z_m = {}
        f, g = self.sde(time)
        mean, std = self.marg_prob(time, time)

        for i, mod in enumerate(mods_list):
            x_mod = data[mod]
            z = torch.randn_like(x_mod).to(self.device)
            mean_i, std_i = self.marg_prob(
                t[:, i].view(x_mod.shape[0], 1), x_mod)
            z_m[mod] = z
            x_t_m[mod] = mean_i * x_mod + std_i * z

        return x_t_m, z_m, mean, std, f, g

    def expand_mask(self, mask, mods_sizes):

        mask = torch.cat([
            mask[:, i].view(mask.shape[0], 1).expand(mask.shape[0], size) for i, size in enumerate(mods_sizes)
        ], dim=1
        )
        return mask

    def train_step(self, data, score_net, eps=1e-3):

        x = concat_vect(data)

        mods_list = list(data.keys())

        mods_sizes = [data[key].size(1) for key in mods_list]

        nb_mods = len(mods_list)
        bs = data[mods_list[0]].size(0)

        if self.importance_sampling:
            t = (self.sample_debiasing_t(
                shape=(x.shape[0], 1))).to(self.device)
        else:
            t = ((self.T - eps) *
                 torch.rand((x.shape[0], 1)) + eps).to(self.device)

        t_n = t.expand((x.shape[0], nb_mods))
        i = torch.randint(low=1, high=len(self.subsets)+1, size=(bs,)) - 1
        mask = self.subsets[i.long(), :]

        mask_time_marg = (mask < 0).int()

        mask_time_cond = mask.clip(0, 1)

        t_n = t_n * mask_time_cond

        x_t_m, z_m, mean, std, f, g = self.sample(
            t=t_n, time=t, data=data, mods_list=mods_list)

        X_t = concat_vect(x_t_m).float()

        t_n = t_n + (self.margin_time) * mask_time_marg
        mask_time_marg = self.expand_mask(mask_time_marg, mods_sizes)

        X_t = X_t * (1 - mask_time_marg)
        if self.fill_zeros == False:
            X_t = X_t + mask_time_marg * torch.randn_like(X_t)

        score = score_net(X_t , 
                          t_n=t_n.float(),
                          t= t,
                        mask= mask, 
                        std = None)

        Z = concat_vect(z_m)

        mask_data_diff = self.expand_mask(mask_time_cond, mods_sizes)

        score = mask_data_diff * score
        Z = mask_data_diff * Z

        total_size = score.size(1)

        weight = (( ( total_size - torch.sum(mask_data_diff,dim=1) ) / total_size ) + 1 ).view(bs,1)

        loss = (weight * torch.square(score - Z)).sum(1, keepdim=False)

        return loss

    def sample_debiasing_t(self, shape):
        """
        non-uniform sampling of t to debias the weight std^2/g^2
        the sampling distribution is proportional to g^2/std^2 for t >= t_epsilon
        for t < t_epsilon, it's truncated
        """
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, t_epsilon=self.t_epsilon, T=self.T)
