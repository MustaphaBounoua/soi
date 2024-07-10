

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from src.baseline.TC_estimation.mi_estimators import  CLUBMean, MINE, NWJ, InfoNCE



### Code reported a modfied from https://github.com/Linear95/TC-estimation 
###



MI_CLASS={
    'CLUB': CLUBMean,
    'MINE': MINE,
    'NWJ': NWJ,
    'InfoNCE': InfoNCE,
    }




class O_Estimator(pl.LightningModule):
    def __init__(self, args,
                 nb_var,
                 test_samples = None,
                 gt= None):  
        '''
        Calculate S-information Estimation for variable X1, X2,..., Xn, each Xi dimension = dim_i
        args:
            dims: a list of variable dimensions, [dim_1, dim_2,..., dim_n]
            hidden_size: hidden_size of vairiational MI estimators
            mi_estimator: the used MI estimator, selected from MI_CLASS
        '''
        super().__init__()
        self.args = args
        
        if args.dim == 1:
            hidden_size = 8
        elif args.dim <= 10:
            hidden_size = 16
        else:
            hidden_size = 20


        self.dims =dims= [args.dim for i in range(nb_var)]
        self.mi_est_type = MI_CLASS[args.mi_e]
        self.test_samples = test_samples
        self.gt = gt

   
        ## Mine diverges using large layers.
        if args.mi_e=="MINE":
            max_dim= 32
        else:
            max_dim = 256
        ## Create the MI estimators for each term in eq.1
        mi_estimator_list_s = [
            self.mi_est_type(
                x_dim=sum(dims) - dim,
                y_dim=dim,
                hidden_size= min( hidden_size * (len(dims)-1), max_dim) 
            )
            for i, dim in enumerate(dims)
        ]
        ## Create the MI estimators for each term in eq.16
        mi_estimator_list_tc = [
            self.mi_est_type(
                x_dim=sum(dims[:i+1]),
                y_dim=dim,
                hidden_size= min(  hidden_size * (1 + i) , max_dim)
            )
            for i, dim in enumerate(dims[:-1])
        ]
        
        self.mi_estimator_list_tc = nn.ModuleList(mi_estimator_list_tc)
        self.mi_estimator_list_s = nn.ModuleList(mi_estimator_list_s)
        self.automatic_optimization = False
    
    def training_step(self, batch, batch_idx):
       
        self.train()
        optimizer = self.optimizers()
        samples = [batch[key].float() .to(self.device) for key in batch.keys()]

        model_loss_t = self.learning_loss_tc(samples)
        model_loss_s = self.learning_loss_s(samples)
        
        loss =torch.stack(model_loss_t + model_loss_s).mean()
        
        self.log("train/loss_t",torch.stack(model_loss_t).mean() )
        self.log("train/model_loss_s",torch.stack(model_loss_s).mean() )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
    def validation_step(self, batch, batch_idx):
       
        self.eval()
        samples = [batch[key].float() .to(self.device) for key in batch.keys()]
        with torch.no_grad():
            model_loss_t = self.learning_loss_tc(samples)
            model_loss_s = self.learning_loss_s(samples)
                        
      
        self.log("test/loss_t",torch.stack(model_loss_t).mean() )
        self.log("test/model_loss_s",torch.stack(model_loss_s).mean() )
        
        return {"loss":torch.stack(model_loss_t + model_loss_s).mean() }
    
    
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if self.current_epoch % self.args.test_epoch == 0 :
            r=self.forward(self.test_samples)
         
            for met in ["tc","dtc","o_inf","s_inf"]:
                self.logger.experiment.add_scalars('Measures/{}'.format(met),  
                                                {'gt': self.gt[met] , 
                                                    'e': r[met] 
                                                    }, global_step=self.global_step)
                
            print("GT: ", np.round( self.gt["o_inf"], decimals=3 ) if self.gt != None else 
                  "Not given", "LINE_{}_estimate: ".format(self.args.mi_e),np.round( r["o_inf"], decimals=3 ) )

        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.args.lr  )
        return optimizer
            
            
    def forward(self, t_dataloader): # samples is a list of tensors with shape [Tensor([batch, dim_i])]
        '''
        forward the estimated total correlation value with the given samples.
        '''
        self.eval()
        s = []
        tc = []
        for batch in tqdm(t_dataloader):
            samples = [batch[key].float() .to(self.device) for key in batch.keys()]
            outputs = []
            for i,_ in enumerate(self.dims):
                x_i=samples[i]
                x_islash =samples[:i] + samples[i+1:]
                cat_sample = torch.cat(x_islash, dim=1)
                with torch.no_grad():
                    outputs.append(self.mi_estimator_list_s[i](cat_sample, x_i).detach())
                
                
            s.append(torch.stack(outputs).sum().cpu().numpy().item())
        
            outputs = []
        
            concat_samples = [samples[0]]
            for i, dim in enumerate(self.dims[1:]):
                cat_sample = torch.cat(concat_samples, dim=1)
                with torch.no_grad():
                    outputs.append(self.mi_estimator_list_tc[i](cat_sample, samples[i+1]).detach())
             
                concat_samples.append(samples[i+1])
            
            tc.append( torch.stack(outputs).sum().cpu().numpy().item() )
            
            torch.cuda.empty_cache()
            
        tc_mean =  np.mean(tc)
        s_mean =   np.mean(s)
        ## S = TC + DTC, DTC = S - TC
        ## O = TC - DTC = 2*TC - S
        return {"tc":tc_mean, "dtc": s_mean - tc_mean, "o_inf":2* tc_mean - s_mean, "s_inf":s_mean}
    
    

    def learning_loss_s(self, samples):
        '''
        return the learning loss to train the parameters of mi estimators.
        '''
        outputs = []
        for i,_ in enumerate(self.dims):
            
            x_i=samples[i]
            x_islash =samples[:i] + samples[i+1:]
            cat_sample = torch.cat(x_islash, dim=1)
            outputs.append(self.mi_estimator_list_s[i].learning_loss(cat_sample, x_i))

        return outputs
    
    
    def learning_loss_tc(self, samples):
        '''
        return the learning loss to train the parameters of mi estimators.
        '''
        outputs = []
        concat_samples = [samples[0]]
        for i, dim in enumerate(self.dims[1:]):
            cat_sample = torch.cat(concat_samples, dim=1)
            outputs.append(self.mi_estimator_list_tc[i].learning_loss(cat_sample, samples[i+1]))
            concat_samples.append(samples[i+1])

        return outputs
    
    def fit(self,train_loader,test_loader):
        args = self.args
        CHECKPOINT_DIR = "{}/baseline_{}/{}/{}/seed_{}/setting_{}/dim_{}/rho_{}/".format(args.out_dir,args.mi_e,args.benchmark,
        args.transformation, args.seed, args.setting, args.dim, args.rho)
    
        trainer = pl.Trainer(logger=TensorBoardLogger(
        save_dir=CHECKPOINT_DIR, name="O_inf"),
                         accelerator='gpu',
                         devices=1,
                         check_val_every_n_epoch=60,
                         max_epochs=args.max_epochs,
                         default_root_dir=CHECKPOINT_DIR,
                         )
        trainer.fit(model=self, train_dataloaders=train_loader,
                val_dataloaders=test_loader)