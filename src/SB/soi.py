import numpy as np
import torch
import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
# from src.model.score_net import UnetMLP
from ..libs.ema import EMA
from ..libs.SDE import VP_SDE
from ..libs.importance import get_normalizing_constant
from ..libs.util import concat_vect, deconcat
from ..models.mlp import UnetMLP_simple
from ..libs.info_measures import minus_x_data, get_tc, pop_elem_i, get_joint_entropy, get_marg_entropy, get_dtc, infer_all_measures_2
from ..models.transformer import DiT


def marginalize_data(x_t, mod, fill_zeros=False):
    x = x_t.copy()
    for k in x.keys():
        if k != mod:
            if fill_zeros:
                x[k] = torch.zeros_like(x_t[k])
            else:
                x[k] = torch.randn_like(x_t[k])
    return x


def cond_x_data(x_t, data, mod):
    x = x_t.copy()
    for k in x.keys():
        if k != mod:
            x[k] = data[k]
    return x


class SOI(pl.LightningModule):

    def __init__(self,
                 lr=1e-3,
                 mod_list={"x": 1, "y": 1, "m": 1},
                 debias=False,
                 weighted=False,
                 use_ema=False,
                 weight_subsets=True,
                 test_samples=None,
                 test_epoch=25,
                 gt=None,
                 debug=False,
                 fill_zeros=False,
                 scores_order=0,
                 batch_size=64,
                 margin_time=1,
                 use_only_margin=False,
                 tx=False
                 ):

        super(SOI, self).__init__()
        self.mod_list = mod_list
        self.gt = gt
        self.debug = debug
        self.weighted = weighted
        self.test_epoch = test_epoch
        self.margin_time = margin_time
        self.tx = tx

        self.sizes = [mod_list[key] for key in mod_list.keys()]
        dim = np.sum(self.sizes)

        if dim <= 10:
            hidden_dim = 128
        elif dim <= 50:
            hidden_dim = 128
        elif dim <= 100:
            hidden_dim = 192
        else:
            hidden_dim = 256

        dim_m = np.max(self.sizes)
        if dim_m <= 5:
            htx = 12
        elif dim_m <= 10:
            htx = 18
        else:
            htx = 24

        time_dim = hidden_dim

        # self.score = UnetMLP(dim= dim , init_dim= hidden_dim ,dim_mults=[]  , time_dim= time_dim ,nb_mod= len(mod_list.keys()) )

        if self.tx==False:
            self.score = UnetMLP_simple(dim=dim, init_dim=hidden_dim, dim_mults=[], time_dim=time_dim,
                                    nb_mod=len(mod_list.keys()))
        else:
            self.score = DiT(depth=4,
                             hidden_size=htx * len(mod_list),
                             mod_sizes=self.sizes,
                             mod_list=list(mod_list.keys()),
                             num_heads=6,
                             variable_input=False,
                             )
        
            

        # self.score = UnetMLP_s(dim= dim , init_dim= hidden_dim, bottelneck= hidden_dim, time_dim= time_dim ,nb_mod= len(mod_list.keys()) )

        self.debias = debias
        self.lr = lr
        self.weight_subsets = weight_subsets
        self.use_ema = use_ema
        self.gt = gt
        self.fill_zeros = fill_zeros
        self.scores_order = scores_order
        self.save_hyperparameters(
            "weight_subsets", "debias", "lr", "use_ema", "batch_size", "mod_list")

        self.test_samples = test_samples

        self.model_ema = EMA(self.score, decay=0.999) if use_ema else None
        self.sde = VP_SDE(importance_sampling=self.debias,
                          fill_zeros=fill_zeros,
                          nb_mod=len(self.mod_list),
                          weight_subsets=weight_subsets,
                          scores_order=scores_order,
                          margin_time=margin_time)

    def training_step(self, batch, batch_idx):

        self.train()

        # forward and compute loss
        loss = self.sde.train_step(batch, self.score).mean()

        self.log("loss", loss)

        return {"loss": loss}

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update(self.score)

    def score_inference(self, x, t=None, t_n=None, mask=None, std=None):
        with torch.no_grad():
            self.eval()
            if self.use_ema:
                self.model_ema.module.eval()
                return self.model_ema.module(x, t=t, t_n=t_n, mask=mask, std=std)
            else:
                return self.score(x, t=t, t_n=t_n, mask=mask, std=std)

    def validation_step(self, batch, batch_idx):
        self.eval()

        # # forward and compute loss
        loss = self.sde.train_step(batch, self.score).mean()
        self.log("loss_test", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.score.parameters(), lr=self.lr)
        return optimizer

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if self.current_epoch % self.test_epoch == 0 and self.current_epoch != 0:

            if self.debug == False:
                print("okay")
                r = self.compute_o_inf_batch(
                    test_loader=self.test_samples, debias=True, nb_iter=10)
                r_uni = self.compute_o_inf_batch(
                    test_loader=self.test_samples, debias=False, nb_iter=10)
            else:
                r = self.compute_o_inf(
                    data=self.test_samples, debias=True, nb_iter=10)
                r_uni = self.compute_o_inf(
                    data=self.test_samples, debias=False, nb_iter=10)

            for met in ["tc", "o_inf", "dtc"]:  # , "s_inf"]:
                self.logger.experiment.add_scalars('Measures/{}'.format(met),
                                                   {'gt': self.gt[met],
                                                    'e': r["simple"][met],

                                                    'ue': r_uni["simple"][met],

                                                    }, global_step=self.global_step)

            if self.debug:
                for i in range(len(self.mod_list)):

                    self.logger.experiment.add_scalars('Debbug/e_j_i{}'.format(i),
                                                       {'gt': self.gt["e_j_i"][i],
                                                        'e': r["e_j_i"][i],
                                                        'ue': r_uni["e_j_i"][i],
                                                        }, global_step=self.global_step)

                    self.logger.experiment.add_scalars('Debbug/e_j_cond_slash_{}'.format(i),
                                                       {'gt': self.gt["e_j"] - self.gt["e_j_minus_i"][i],
                                                        'e': r["e_j_cond_slash"][i],
                                                        'ue': r_uni["e_j_cond_slash"][i],
                                                        }, global_step=self.global_step)

                self.logger.experiment.add_scalars('Debbug/e_j',
                                                   {'gt': self.gt["e_j"],
                                                    'e': r["e_j"],
                                                    'ue': r_uni["e_j"],
                                                    }, global_step=self.global_step)

    def compute_o_inf(self, data, debias=False, eps=1e-5, nb_iter=10):

        self.sde.device = self.device
        self.score.eval()

        mods_list = list(data.keys())

        mods_sizes = [data[key].size(1) for key in mods_list]
        data = {k: data[k].to(self.device) for k in mods_list}

        N = len(mods_list)
        M = data["x0"].shape[0]

        e_j_mc = []

        tc_mc = []
        dtc_mc = []

        e_j_i_mc = {mod: [] for mod in data.keys()}
        e_j_cond_slash_mc = {mod: [] for mod in data.keys()}
        for i in range(nb_iter):
            if debias:
                t = (self.sde.sample_debiasing_t(shape=(M, 1))).to(self.device)
            else:
                t = ((self.sde.T - eps) * torch.rand((M, 1)) + eps).to(self.device)

            t_n = t.expand((M, N))
            X_t, z_m, mean, std, f, g = self.sde.sample(
                t_n, t, data, mods_list)

            marg_masks = {}
            cond_x_mask = {}

            for i, mod in enumerate(mods_list):
                mask = [0] * N
                mask[i] = 1
                marg_masks[mod] = torch.tensor(mask).to(
                    self.device).expand(t_n.size())

                mask = [0] * N
                mask[i] = 1
                cond_x_mask[mod] = torch.tensor(mask).to(
                    self.device).expand(t_n.size())

            x_t_joint = concat_vect(X_t)

            marginals = {mod: concat_vect(marginalize_data(
                X_t, mod, fill_zeros=self.fill_zeros)) for mod in mods_list}

            cond_x = {mod: concat_vect(cond_x_data(X_t, data, mod))
                      for mod in mods_list}

            if debias:
                std_w = None
            else:
                std_w = std

            with torch.no_grad():

                s_joint = - self.score_inference(x_t_joint.float(), t=t,
                                                 t_n=t_n,
                                                 std=std_w,
                                                 mask=torch.ones_like(marg_masks[mods_list[0]])).detach()
                s_marg = {}
                s_cond_x = {}
                for mod in mods_list:

                    s_marg[mod] = - self.score_inference(marginals[mod].float(), t=t,
                                                         mask=marg_masks[mod] +
                                                         (-1) *
                                                         (1 - marg_masks[mod]),
                                                         t_n=t_n * marg_masks[mod] + (self.margin_time) * (1 - marg_masks[mod]), std=std_w).detach()

                    s_cond_x[mod] = - self.score_inference(cond_x[mod].float(), t=t,
                                                           mask=cond_x_mask[mod],
                                                           t_n=t_n * cond_x_mask[mod], std=std_w).detach()

            s_joint = deconcat(s_joint, mods_list, mods_sizes)

            s_marg = {
                mod: deconcat(s_marg[mod], mods_list, mods_sizes)[mod]
                for mod in mods_list
            }

            s_cond_x = {
                mod: deconcat(s_cond_x[mod], mods_list, mods_sizes)[mod]
                for mod in mods_list
            }

            if self.debug:
                e_j = get_joint_entropy(self.sde, s_joint=s_joint,
                                        x_t=X_t, std=std, g=g,
                                        debias=debias,
                                        x_0=data,
                                        mean=mean)
                e_j_mc.append(e_j)

            tc = get_tc(self.sde, s_joint=s_joint, s_marg=s_marg,
                        x_t=X_t, std=std, g=g,
                        debias=debias,
                        mean=mean)

            dtc = get_dtc(self.sde, s_joint=s_joint, s_cond=s_cond_x,
                          x_t=X_t, std=std, g=g,
                          debias=debias,
                          mean=mean)
            tc_mc.append(tc)
            dtc_mc.append(dtc)

            # e_j_i = []
            # e_j_cond_slash = []
            if self.debug:

                for index, mod_minus_i in enumerate(mods_list):
                    cond_e = get_marg_entropy(self.sde, s_joint=s_cond_x[mod_minus_i],
                                              x_t=X_t[mod_minus_i],
                                              x_0=data[mod_minus_i],
                                              std=std, g=g,
                                              debias=debias,
                                              mean=mean)

                    e_j_cond_slash_mc[mod_minus_i].append(cond_e)

                    marg_e = get_marg_entropy(self.sde, s_joint=s_marg[mod_minus_i],
                                              x_t=X_t[mod_minus_i],
                                              x_0=data[mod_minus_i],
                                              std=std, g=g,
                                              debias=debias,
                                              mean=mean)

                    e_j_i_mc[mod_minus_i].append(marg_e)

        tc = {"simple": torch.stack([t["simple"] for t in tc_mc]).mean(),
              }

        dtc = {"simple": torch.stack([t["simple"] for t in dtc_mc]).mean(),
               }

        r = {
            "simple":  infer_all_measures_2(tc, dtc, type_="simple"),

        }

        if self.debug:
            e_j = torch.stack(e_j_mc).mean().item()
            e_j_i = [torch.stack(e_j_i_mc[key]).mean().item()
                     for key in e_j_i_mc.keys()]

            e_j_cond_slash = [torch.stack(e_j_cond_slash_mc[key]).mean(
            ).item() for key in e_j_cond_slash_mc.keys()]

            r["e_j"] = e_j
            r["e_j_i"] = e_j_i
            r["e_j_cond_slash"] = e_j_cond_slash
        return r

    def compute_o_inf_batch(self, test_loader, debias=False, eps=1e-5,  nb_iter=10):

        mets = ["o_inf", "s_inf", "tc", "dtc"]
        out = {
            "simple": {met: [] for met in mets},
        }

        for batch in tqdm(test_loader):
            r = self.compute_o_inf(batch, debias=debias, nb_iter=nb_iter)
            for met in mets:
                out["simple"][met].append(r["simple"][met])

        for met in mets:
            out["simple"][met] = torch.stack(
                out["simple"][met]).mean().cpu().numpy().item()
        return out
