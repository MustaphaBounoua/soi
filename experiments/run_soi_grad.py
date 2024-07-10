

import json
import os
import torch
import pytorch_lightning as pl
from src.libs.soi_grad import SOI_grad
from src.libs.util import get_samples
from experiments.config import get_config
from src.benchmark.synthetic_dataset import get_task,get_dataloader


parser = get_config()


def benchmark_exp(args):

  
    task = get_task(args)        
    print(task.o_inf())
    device="cuda" if args.accelerator == "gpu" else "cpu"
    print(torch.cuda.is_available())
    train_loader, test_loader = get_dataloader(task, args)
    test_samples = get_samples(test_loader,device , N=10000)
    args.hidden_dim = None
    model = SOI_grad(args, test_samples=test_samples,gt=task.get_summary(),  nb_var=task.nb_var)

    model.fit(train_loader, test_loader)
    model.to(device)
    model.eval()
    
    results = model.compute_o_inf_with_grad(test_samples)
    return {args.rho:{"gt": task.get_summary(),
                      "e":results}}


if __name__ == "__main__":

    

    args = parser.parse_args()
    
    pl.seed_everything(args.seed)

    r = benchmark_exp(args)

    path = "{}/soi_grad/{}/{}/{}/seed_{}/setting_{}/dim_{}/".format(args.results_dir,args.arch ,
                                                               args.benchmark,args.transformation,
                                                                  args.seed, args.setting, args.dim)
    isExist = os.path.exists(path)
    if not os.path.exists(path):
        os.makedirs(path)

    path = path + "/rho{}.json".format(args.rho)
    with open(path, 'w') as f:
        json.dump(r, f)
