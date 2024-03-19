

import sys
import json
import argparse
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from src.util import get_samples
from src.SB.soi_vbn import SOI
from src.vbn.vbn import VBNDataset

parser = argparse.ArgumentParser()

parser.add_argument('--time_bin', type=int, default=0)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--fill_zeros', type=int, default=0)
parser.add_argument('--scores_order', type=int, default=1)
parser.add_argument('--nb_epoch', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--weight_subsets', type=int, default=1)
parser.add_argument('--test_epoch', type=int, default=2)
parser.add_argument('--change', type=str, default="change")
parser.add_argument('--setting', type=int, default=3)
parser.add_argument('--dim', type=int, default=10)




def benchmark_exp(args):


    seed= args.seed
    fill_zeros = args.fill_zeros ==1
    scores_order = args.scores_order
    
    lr = args.lr
    bs = args.bs
    weight_subsets = args.weight_subsets ==1
    test_epoch = args.test_epoch
    dim = args.dim
    
    r = {} 
    if args.setting == 3:
        structure = ['VISp', 'VISl', 'VISal']
    else:
        structure = ["VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm" ]
    
    
    if dim==10:
        file_path= "/home/****/work/gsoi/vbn5/"
    elif dim == 25:
        file_path= "/home/****/work/gsoi/vbn02/"
    elif dim == 50:
        file_path= "/home/****/work/gsoi/vbn001/" 
        
    train_set = VBNDataset(structures=structure,
                           change = args.change, 
                           preprocess= True,
                           size = None,
                           file_path= file_path,
                           time_bin= args.time_bin,
                           aggregate=True ,
                           nn_c = dim)
    
    print(len(train_set))
    
    train_loader = DataLoader(train_set, batch_size=bs,shuffle=True, #pin_memory=True,
                                    num_workers=8, drop_last=True)



    mod_list={ i : dim for i in structure }

  

    model = SOI(debias=True,
                        test_samples= train_loader,
                        gt = 0.0,
                        lr=lr,
                        mod_list =mod_list ,
                        use_ema=True, 
                        batch_size = bs,
                        debug=False,
                        margin_time= 1,
                        scores_order= scores_order,
                        fill_zeros= False,
                        weight_subsets=weight_subsets,
                        test_epoch = 100)

    CHECKPOINT_DIR = "trained_models/vbn/dim{}/setting_{}/seed_{}/bin{}/{}".format(args.dim,args.setting,args.seed,args.time_bin,args.change)
    
    tb_logger =  TensorBoardLogger(save_dir = CHECKPOINT_DIR,name="vbn")
    trainer = pl.Trainer( logger= tb_logger,
                            accelerator='gpu', 
                            devices= 1,check_val_every_n_epoch=50,
                            max_epochs= args.nb_epoch, 
                            default_root_dir = CHECKPOINT_DIR,
                            )
    trainer.fit(model=model, train_dataloaders=train_loader  )
    r = {}
    
    model.eval()
    model.to("cuda")
    r["e"] = model.compute_o_inf_batch(train_loader, debias= True,nb_iter=10)
    
    sessions=train_set.get_sessions()
    
    r_s ={"o_inf":[],"s_inf":[],"tc":[],"dtc":[]}
    for sess in sessions:
        out=model.compute_o_inf(sess, debias= True,nb_iter=10)
        r_s["o_inf"].append(out["simple"]["o_inf"].item())
        r_s["s_inf"].append(out["simple"]["s_inf"].item())
        r_s["tc"].append(out["simple"]["tc"].item())
        r_s["dtc"].append(out["simple"]["dtc"].item())
    r["ses"]= r_s
    return r
        


if __name__ =="__main__":
    
    print(torch.cuda.is_available())
    
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    
    r = benchmark_exp(args)
    
    
    path = "results/vbn/dim_{}/setting_{}/seed_{}/bin{}/".format(args.dim,args.setting,args.seed,args.time_bin)
    isExist = os.path.exists(path)
    if not os.path.exists(path):
        os.makedirs(path)
        
    path = path + "/results_{}.json".format(args.change)
    with open(path, 'w') as f:
        json.dump(r, f)