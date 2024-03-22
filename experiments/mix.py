

import json
import argparse
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger




from src.SB.soi import SOI
from src.benchmark.toy_dataset import Task_redundant ,Task_combination,Task_synergy


parser = argparse.ArgumentParser()

parser.add_argument('--rho', type=float, default=0.1 )
parser.add_argument('--seed', type=int, default=3)

parser.add_argument('--setting', type=int, default=0 )
parser.add_argument('--dim', type=int, default=1 )

parser.add_argument('--N', type=int, default=100*1000)
parser.add_argument('--fill_zeros', type=int, default=1)
parser.add_argument('--scores_order', type=int, default=1)
parser.add_argument('--nb_epoch', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--bs', type=int, default=256)
parser.add_argument('--weight_subsets', type=int, default=1)
parser.add_argument('--test_epoch', type=int, default=5000)
parser.add_argument('--trans', type=str, default="")




SETTINGS = [ {"red": [3,4],"syn": [3]}]
RHO = 0.7








def benchmark_exp(args):
    rho = args.rho
    
    dim= args.dim
    N = args.N
    seed= args.seed
    fill_zeros = args.fill_zeros ==1
    scores_order = args.scores_order
    nb_epoch = args.nb_epoch
    lr = args.lr
    bs = args.bs
    weight_subsets = args.weight_subsets ==1
    test_epoch = args.test_epoch
    setting = SETTINGS[args.setting]
    
    
    r = {} 
    

    
    tasks =[]
    for i in setting["red"]:
        tasks.append(Task_redundant(nb_var=i, rho=rho ,dim = dim))
        
    for i in setting["syn"]:
        tasks.append(Task_synergy(nb_var=i, rho=RHO ,dim = dim))
    
    task = Task_combination(tasks=tasks ,dim=dim,transformation= args.trans)
    
    nb_mod  = task.nb_var
    
    print( task.o_inf()[0]  )
        
    if args.trans == "":
        rescale = False
    else:
        rescale = True

    d_train, d_test = task.get_torch_dataset(N,10*1000,dim=dim,rescale=rescale,seed= seed)

    train_loader = DataLoader(d_train, batch_size=bs,shuffle=True,
                                    num_workers=8, drop_last=True)

    test_loader = DataLoader(d_test, batch_size=500,
                                    shuffle= False,
                                    num_workers=4, drop_last=False)

    mod_list={ "x"+ str(i) : dim for i in range(nb_mod) }

   
    model = SOI(debias=True,
                        test_samples= test_loader,
                        gt = task.get_summary(),
                        lr = lr,
                        mod_list = mod_list ,debug =False,
                        use_ema = True, 
                        margin_time= 1,
                        use_only_margin=False,
                        batch_size = bs,
                        scores_order= scores_order,
                        fill_zeros= False,
                        weight_subsets=weight_subsets,
                        test_epoch = test_epoch)

    CHECKPOINT_DIR = "trained_models/soi/mix_{}/seed_{}/setting_{}/dim_{}/rho_{}/".format(args.trans,args.seed,args.setting,args.dim,args.rho)
    
    tb_logger =  TensorBoardLogger(save_dir = CHECKPOINT_DIR,name="o_inf_X")
    trainer = pl.Trainer( logger= tb_logger,
                            accelerator='gpu', 
                            devices= 1,check_val_every_n_epoch=50,
                            max_epochs= nb_epoch, 
                            default_root_dir = CHECKPOINT_DIR,
                            )
    trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=test_loader  )
    r[rho] ={} 
    r[rho]  ["gt"] = task.get_summary()
    model.eval()
    model.to("cuda")
    r[rho]  ["e"] =  model.compute_o_inf_batch(model.test_samples, debias= True,nb_iter=10)
    return r
        


if __name__ =="__main__":
    
    print(torch.cuda.is_available())
    
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    
    r = benchmark_exp(args)
    
    
    path = "results/soi_{}/mix/seed_{}/setting_{}/dim_{}/".format(args.trans,args.seed,args.setting,args.dim)
    isExist = os.path.exists(path)
    if not os.path.exists(path):
        os.makedirs(path)
        
    path = path + "/rho{}.json".format(args.rho)
    with open(path, 'w') as f:
        json.dump(r, f)