import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from .SDE import VP_SDE
from .util import EMA,concat_vect, deconcat, marginalize_data, cond_x_data , get_samples
from .info_measures import get_tc, get_dtc, compute_all_measures, compute_entropy
from ..models.mlp import UnetMLP_simple
from ..models.transformer import DiT


class SOI(pl.LightningModule):

    def __init__(self,args,nb_var=None,gt=None,var_list = None):
               
                 
        super(SOI, self).__init__()
        if var_list ==None:
            var_list = {"x" + str(i): args.dim for i in range(nb_var)}

        self.args = args
        self.var_list = list(var_list.keys())
        self.sizes = list(var_list.values())
        self.gt = gt
        
        self.marginals = -1 if self.args.arch == "tx" else 1 # How to handle the marginals scores, non marginals are set to null vector zero if transfomer or gaussian noise if MLP
       
        self.save_hyperparameters("args")

        
        if hasattr(args, 'hidden_dim')==False or args.hidden_dim == None:
            hidden_dim = self.calculate_hidden_dim()
        else:
            hidden_dim = args.hidden_dim
        
        if self.args.arch == "mlp":
            self.score = UnetMLP_simple(dim=np.sum(self.sizes), init_dim=hidden_dim, dim_mults=[],
                                        time_dim=hidden_dim, nb_var=len(var_list.keys()))
        elif self.args.arch == "tx":
            self.score = DiT(depth=4,hidden_size=hidden_dim , var_list=var_list)
        
        
        self.model_ema = EMA(self.score, decay=0.999) if self.args.use_ema else None

        self.sde = VP_SDE(importance_sampling=self.args.importance_sampling,
                          var_sizes=self.sizes,
                          weight_s_functions=self.args.weight_s_functions,
                          o_inf_order=self.args.o_inf_order,
                          marginals=self.marginals
                          )
    
    
    def fit(self,train_loader,test_loader):
        if test_loader ==None:
            test_loader = train_loader ## train and test on the same dataset
        
        self.test_samples = get_samples(test_loader,device="cuda"if self.args.accelerator == "gpu" else "cpu")
        args = self.args
        CHECKPOINT_DIR = "{}/Soi{}/{}/{}/seed_{}/setting_{}/dim_{}/rho_{}/".format(args.out_dir,"_grad" if args.o_inf_order>1 else "",
                                                                                   args.benchmark,
        args.transformation, args.seed, args.setting, args.dim, args.rho)
        trainer = pl.Trainer(logger=pl.loggers.TensorBoardLogger(save_dir=CHECKPOINT_DIR, name="o_inf"),
                         default_root_dir=CHECKPOINT_DIR,
                         accelerator=self.args.accelerator,
                         devices=self.args.devices,
                         max_epochs=self.args.max_epochs,
                         check_val_every_n_epoch=self.args.check_val_every_n_epoch)  
    
        trainer.fit(model=self, train_dataloaders=train_loader,
                val_dataloaders=test_loader)
    
    
    def on_fit_start(self):
        self.sde.set_device(self.device) 
        
    def training_step(self, batch, batch_idx):
        self.train()
        loss = self.sde.train_step(batch, self.score_forward).mean()
        self.log("loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        self.eval()
        loss = self.sde.train_step(batch, self.score_forward).mean()
        self.log("loss_test", loss)
        return {"loss": loss}

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update(self.score)

    
    
    
    def score_forward(self, x, t=None, mask=None, std=None):
        """
        Perform score inference on the input data.

        Args:
            x (torch.Tensor): Concatenated variables.
            t (torch.Tensor, optional): The time t. 
            mask (torch.Tensor, optional): The mask data.
            std (torch.Tensor, optional): The standard deviation to rescale the network output.

        Returns:
            torch.Tensor: The output score function (noise/std) if std !=None , else return noise .
        """
        if self.args.arch == "tx":
            return self.score(x, t=t, mask=mask, std=std)
        else:
            # MLP network requires the multitime vector
            t = t.expand(mask.size()) * mask.clip(0, 1)
            marg = (mask < 0).int()
            t = t * (1 - marg) + self.marginals * marg
            return self.score(x, t=t, std=std)

    def score_inference(self, x, t=None, mask=None, std=None):
        """
        Perform score inference on the input data.

        Args:
            x (torch.Tensor): Concatenated variables.
            t (torch.Tensor, optional): The time t. 
            mask (torch.Tensor, optional): The mask data.
            std (torch.Tensor, optional): The standard deviation to rescale the network output.

        Returns:
            torch.Tensor: The output score function (noise/std) if std !=None , else return noise .
        """
        # Get the model to use for inference, use the ema model if use_ema is set to True

        score = self.model_ema.module if self.args.use_ema else self.score
        with torch.no_grad():
            score.eval()
            if self.args.arch == "tx":
                mask = mask.view(1, len(mask)).expand(x.size(0), len(mask))
                return score(x, t=t, mask=mask, std=std)
            else:
                t = t.expand((x.size(0), mask.size(-1))) * mask.clip(0, 1)
                marg = (mask < 0).int()
                t = t * (1 - marg) + self.marginals * marg
                return score(x, t=t, std=std)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.score.parameters(), lr=self.args.lr)
        return optimizer

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if self.current_epoch % self.args.test_epoch == 0 and self.current_epoch != 0 and self.current_epoch> self.args.warmup_epochs:
            self.logger_estimates()

    def infer_scores_for_o_inf(self,x_t,t, data_0, std_w,marg_masks,cond_mask):
        
        
        
        X_t = deconcat(x_t, self.var_list, self.sizes)

        # Prepare the marginals input by replacing the non marginals with zeros or gaussian noise
        # marginals= -1 means that the non marginals are replaced with zeros, gaussian noise otherwise
        marginals = {x_i: concat_vect(marginalize_data(
                X_t, x_i, fill_zeros=(self.marginals == -1))) for x_i in self.var_list}

        # Prepare the conditional input by replacing the conditional with clean samples
        cond_x = {x_i: concat_vect(cond_x_data(X_t, data_0, x_i))
                      for x_i in self.var_list}
        # Don't scale by std if importance sampling it will be incorporated in the normalized constante
        

        # Compute the denoising score functions
        with torch.no_grad():
            s_joint = - self.score_inference( x_t, t=t, std=std_w, mask=torch.ones_like(marg_masks[self.var_list[0]])).detach()
            s_marg = {}
            s_cond_x = {}

            for x_i in self.var_list:
                s_marg[x_i] = - self.score_inference(
                        marginals[x_i], t=t, mask=marg_masks[x_i], std=std_w).detach()
                s_cond_x[x_i] = - self.score_inference(
                        cond_x[x_i], t=t, mask=cond_mask[x_i], std=std_w).detach()

        s_joint = deconcat(s_joint, self.var_list, self.sizes)
        s_marg = {x_i: deconcat(s_marg[x_i], self.var_list, self.sizes)[x_i]
                      for x_i in self.var_list}

        s_cond_x = {x_i: deconcat(s_cond_x[x_i], self.var_list, self.sizes)[x_i]
                        for x_i in self.var_list}
        return s_joint, s_marg, s_cond_x



    def compute_o_inf(self, data=None, eps=1e-5):
        """
        Compute the o_inf.

        Args:
            data (dict): A dictionary containing the input data.{x0:  , x1: , x2: , ...}
            importance_sampling (bool, optional): Flag indicating whether to use importance sampling. Defaults to False.
            eps (float, optional): A small value used for sampling if importance sampling is set to false. Defaults to 1e-5.
            nb_iter (int, optional): Number of iterations for Monte Carlo estimation. Defaults to 10.

        Returns:
            tuple: A tuple containing the computed o_inf measures.

        """
        self.eval()
        self.to("cuda" if self.args.accelerator == "gpu" else "cpu")
        if data==None:
            data = self.test_samples
            
        self.sde.device = self.device
        var_list = list(data.keys())
        data_0 = {x_i: data[x_i].to(self.device) for x_i in var_list}
        x_0 = concat_vect(data_0)

        N = len(self.sizes)
        M = x_0.shape[0]

        tc_mc = []
        dtc_mc = []
        marg_masks, cond_mask = self.get_masks(var_list)

        for i in range(self.args.mc_iter):
            # Sample t
            if self.args.importance_sampling:
                t = (self.sde.sample_importance_sampling_t(
                    shape=(M, 1))).to(self.device)
            else:
                t = ((self.sde.T - eps) * torch.rand((M, 1)) + eps).to(self.device)
            _, g = self.sde.sde(t)
            # Sample from the SDE (pertrbe the data with noise at time)
            x_t, _, _, std = self.sde.sample(x_0=x_0, t=t)
            
            std_w = None if self.args.importance_sampling else std
            
            s_joint, s_marg, s_cond_x = self.infer_scores_for_o_inf(x_t,t, data_0, std_w, marg_masks, cond_mask)

            tc_mc.append(get_tc(self.sde, s_joint=s_joint, s_marg=s_marg, g=g,
                                importance_sampling=self.args.importance_sampling,))

            dtc_mc.append(get_dtc(self.sde, s_joint=s_joint, s_cond=s_cond_x, g=g,
                                  importance_sampling=self.args.importance_sampling))

        tc, dtc = np.mean(tc_mc), np.mean(dtc_mc)

        return compute_all_measures(tc, dtc)
    

    def get_masks(self, var_list):
        """_summary_
        Returns:
            dict , dict :  marginal masks: {x_i: [-1,...,1,...,-1]
            } , conditional masks: {x_i: [0,...,1,...,0]
        """
        masks = {}
        N = len(var_list)
        for i, x_i in enumerate(var_list):
            mask = [0] * N
            mask[i] = 1
            masks[x_i] = torch.tensor(mask).to(self.device)
        return {k: masks[k]*2 - 1 for k in masks}, masks



    def compute_entropies(self, data, eps=1e-5):
        """
        Compute the entropies using the score functions as done in [1]:

            [1] "Franzese, G., Bounoua, M., & Michiardi, P. (2023). 
            MINDE: Mutual Information Neural Diffusion Estimation. arXiv preprint arXiv:2310.09031."
        Args:
            data (dict): A dictionary containing the input data.{x0:  , x1: , x2: , ...}
            importance_sampling (bool, optional): Flag indicating whether to use importance sampling. Defaults to False.
            eps (float, optional): A small value used for sampling if importance sampling is set to false. Defaults to 1e-5.
            nb_iter (int, optional): Number of iterations for Monte Carlo estimation. Defaults to 10.

        Returns:
            tuple: A tuple containing the computed o_inf measures.

        """
        var_list = list(data.keys())
        data_0 = data 
        x_0 = concat_vect(data_0)

        N = len(self.sizes)
        M = x_0.shape[0]

        marg_masks, cond_mask = self.get_masks(var_list)
        e_j_i_mc = {x_i: [] for x_i in data_0.keys()}
        e_j_cond_slash_mc = {x_i: [] for x_i in data_0.keys()}
        e_j_mc = []
        for i in range(self.args.mc_iter):
            # Sample t
            if self.args.importance_sampling:
                t = (self.sde.sample_importance_sampling_t(
                    shape=(M, 1))).to(self.device)
            else:
                t = ((self.sde.T - eps) * torch.rand((M, 1)) + eps).to(self.device)
            _, g = self.sde.sde(t)
            # Sample from the SDE (pertrbe the data with noise at time t)
            x_t, _, mean, std = self.sde.sample(x_0=x_0, t=t)
            std_w = None if self.args.importance_sampling else std
            
            s_joint, s_marg, s_cond_x = self.infer_scores_for_o_inf(x_t,t, data_0, std_w, marg_masks, cond_mask)
            
            X_t = deconcat(x_t, self.var_list, self.sizes)
            
            e_j = compute_entropy(self.sde, score=concat_vect(s_joint),
                                    x_t=x_t, std=std, g=g,
                                    importance_sampling=self.args.importance_sampling,
                                    x_0=x_0,
                                    mean=mean)
            e_j_mc.append(e_j)

            for x_minus_i in var_list:
                cond_e = compute_entropy(self.sde, score=s_cond_x[x_minus_i],
                                          x_t=X_t[x_minus_i],
                                          x_0=data_0[x_minus_i],
                                          std=std, g=g,
                                          importance_sampling=self.args.importance_sampling,
                                          mean=mean)

                e_j_cond_slash_mc[x_minus_i].append(cond_e)

                marg_e = compute_entropy(self.sde, score=s_marg[x_minus_i],
                                          x_t=X_t[x_minus_i],
                                          x_0=data_0[x_minus_i],
                                          std=std, g=g,
                                          importance_sampling=self.args.importance_sampling,
                                          mean=mean)

                e_j_i_mc[x_minus_i].append(marg_e)

        e_j = np.mean(e_j_mc)
        e_j_i = [np.mean(e_j_i_mc[key])
                 for key in e_j_i_mc.keys()]
        e_j_cond_slash = [np.mean(e_j_cond_slash_mc[key]) for key in e_j_cond_slash_mc.keys()]

        return {"e_joint": e_j, "e_marg_i": e_j_i, "e_i_cond_slash": e_j_cond_slash}
    

    def compute_o_inf_batch(self, test_loader ):

        # Compute the o_inf when the data is in dataloader format
        
        mets = ["o_inf", "s_inf", "tc", "dtc"]
        out = {
            met: [] for met in mets
        }
        for batch in tqdm(test_loader):
            r = self.compute_o_inf(batch)
            for met in mets:
                out[met].append(r[met])
        return {met: np.mean(out[met]) for met in mets }


    def calculate_hidden_dim(self):
        # return dimensions for the hidden layers
        if self.args.arch == "mlp":
            dim = np.sum(self.sizes)
            if dim <= 10:
                hidden_dim = 128
            elif dim <= 50:
                hidden_dim = 128
            elif dim <= 100:
                hidden_dim = 192
            else:
                hidden_dim = 256
            return hidden_dim
        else:
            dim_m = np.max(self.sizes)
            if dim_m <= 5:
                htx = 60
            if dim_m <= 10:
                htx = 72
            elif dim_m <= 15:
                htx = 96
            else:
                htx = 128
            return htx

    def logger_estimates(self):
        # Log the estimates of the o_inf measures durring training

        
        r = self.compute_o_inf(data=self.test_samples)

        print("Epoch: ",self.current_epoch," GT: ", np.round( self.gt["o_inf"], decimals=3 ) if self.gt != None else "Not given", "SOI_estimate: ",np.round( r["o_inf"], decimals=3 ) )

        for met in ["tc", "o_inf", "dtc"]:  # , "s_inf"]:
                self.logger.experiment.add_scalars('Measures/{}'.format(met),
                                                   {'gt': self.gt[met] if self.gt != None else 0,
                                                    'e': r[met],
                                                    }, global_step=self.global_step)

        if self.args.debug:
                entropies = self.compute_entropies(data=self.test_samples)
                for i in range(len(self.var_list)):

                    self.logger.experiment.add_scalars('Debbug/e_marg_i{}'.format(i),
                                                       {'gt': self.gt["e_marg_i"][i] if self.gt != None else 0,
                                                        'e': entropies["e_marg_i"][i],
                                                        }, global_step=self.global_step)

                    self.logger.experiment.add_scalars('Debbug/e_i_cond_slash{}'.format(i),
                                                       {'gt': self.gt["e_joint"] - self.gt["e_minus_i"][i] if self.gt != None else 0,
                                                        'e': entropies["e_i_cond_slash"][i],
                                                        }, global_step=self.global_step)

                self.logger.experiment.add_scalars('Debbug/e_joint',
                                                   {'gt': self.gt["e_joint"] if self.gt != None else 0,
                                                    'e': entropies["e_joint"],
                                                    }, global_step=self.global_step)
                
