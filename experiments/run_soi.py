

import json
import os
import pytorch_lightning as pl

from src.libs.soi import SOI
from src.libs.util import get_samples
from experiments.config import get_config
from src.benchmark.synthetic_dataset import get_task,get_dataloader

parser = get_config()


def benchmark_exp(args):

    task = get_task(args)

    print("O_inf of this task is : ", task.o_inf())
    device = "cuda" if args.accelerator == "gpu" else "cpu"
    train_loader, test_loader = get_dataloader(task, args)
    test_samples = get_samples(
        test_loader, device=device, N=10000)
    args.hidden_dim = None ##set automotaically in the model class
    model = SOI(args, test_samples=test_samples,
                gt=task.get_summary(), nb_var=task.nb_var)
    model.fit(train_loader, test_loader)
    model.to(device)
    model.eval()

    results = model.compute_o_inf(test_samples)
    
    return {args.rho: {"gt": task.get_summary(),
                       "e": results}}


if __name__ == "__main__":

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    r = benchmark_exp(args)

    path = "{}/soi/{}/{}/{}/seed_{}/setting_{}/dim_{}/".format(args.results_dir, args.arch ,
                                                               args.benchmark, args.transformation,
                                                               args.seed, args.setting, args.dim)
    isExist = os.path.exists(path)
    if not os.path.exists(path):
        os.makedirs(path)

    path = path + "/rho{}.json".format(args.rho)
    with open(path, 'w') as f:
        json.dump(r, f)
