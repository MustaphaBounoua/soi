

import json
import os
import pytorch_lightning as pl
from src.baseline.baseline import O_Estimator
from experiments.config import get_config_baseline
from src.benchmark.synthetic_dataset import get_task, get_dataloader

parser = get_config_baseline()

def benchmark_exp(args):

    task = get_task(args)

    print(task.o_inf())
    summary = task.get_summary()

    train_loader, test_loader = get_dataloader(task,args)

    model = O_Estimator(args = args,nb_var=task.nb_var,
                        test_samples=test_loader,
                        gt=summary,
                        )
    model.fit(train_loader, test_loader)
    model.eval()
    r = {
        args.rho: {
            "gt": task.get_summary(),
            "e": model.forward(model.test_samples),
        }
    }

    return r


if __name__ == "__main__":

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    r = benchmark_exp(args)

    path = "{}/baseline/{}/{}/{}/seed_{}/setting_{}/dim_{}/".format(args.results_dir, args.mi_e,
                                                                    args.benchmark, args.transformation,
                                                                    args.seed, args.setting, args.dim)
    isExist = os.path.exists(path)
    if not os.path.exists(path):
        os.makedirs(path)

    path = path + "/rho{}.json".format(args.rho)
    with open(path, 'w') as f:
        json.dump(r, f)
