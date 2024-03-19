

from src.benchmark.toy_dataset import Task_redundant, Task_combination, Task_synergy
from src.baseline.baseline import O_Estimator
import sys
import json
import argparse
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

root = "/home/****/work/soi/"
sys.path.append(root)


parser = argparse.ArgumentParser()

parser.add_argument('--rho', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--mi_e', type=str, default="MINE")
parser.add_argument('--setting', type=int, default=0)
parser.add_argument('--dim', type=int, default=1)
parser.add_argument('--N', type=int, default=1000 * 100)
parser.add_argument('--nb_epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--trans', type=str, default="")


SETTINGS = [
    [3, 3, 4],
]




def benchmark_exp(args):
    rho = args.rho

    dim = args.dim
    N = args.N
    seed = args.seed

    nb_epoch = args.nb_epoch
    lr = args.lr
    bs = args.bs

    setting = SETTINGS[args.setting]

    task = Task_combination(tasks=[Task_synergy(nb_var=i, rho=rho, dim=dim)
                                   for i in setting], dim=dim, transformation=args.trans, use_det=False)

    nb_mod = task.nb_var

    summ = task.get_summary()
    print(task.o_inf()[0])

    if args.trans == "":
        rescale = False
    else:
        rescale = True

    d_train, d_test = task.get_torch_dataset(N, 10*1000,
                                             dim=dim,
                                             rescale=rescale,
                                             seed=seed)

    train_loader = DataLoader(d_train, batch_size=bs, shuffle=True, pin_memory=True,
                              num_workers=16, drop_last=True)

    test_loader = DataLoader(d_test, batch_size=64,
                             shuffle=False, pin_memory=True,
                             num_workers=16, drop_last=False)

    mod_list = {"x" + str(i): dim for i in range(nb_mod)}

    test_samples = test_loader
    hidden_size = 10
    if dim == 1:
        hidden_size = 8
    elif dim <= 10:
        hidden_size = 16
    else:
        hidden_size = 20

    print("hey")
    model = O_Estimator(
        dims=[dim for i in range(nb_mod)],
        test_samples=test_samples,
        gt=summ,
        hidden_size=hidden_size,
        mi_estimator=args.mi_e,
        lr=lr,
        test_epoch=10,
    )

    CHECKPOINT_DIR = "trained_models/"
    tb_logger = TensorBoardLogger(
        save_dir=CHECKPOINT_DIR, name="baseline_{}/syn/model_{}/seed_{}/setting_{}/dim_{}/rho_{}/"
        .format(args.trans, args.mi_e, args.seed, args.setting, args.dim, args.rho))
    trainer = pl.Trainer(logger=tb_logger,
                         accelerator='gpu',
                         devices=1, check_val_every_n_epoch=60,
                         max_epochs=nb_epoch,
                         default_root_dir=CHECKPOINT_DIR,
                         )
    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=test_loader)
    r = {}
    r[rho] = {}
    r[rho]["gt"] = task.get_summary()
    model.eval()
    model.to("cuda")
    r[rho]["e"] = model.forward(model.test_samples)
    return r


if __name__ == "__main__":

    print(torch.cuda.is_available())

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    r = benchmark_exp(args)

    path = "results/baseline_{}/syn/model_{}/seed_{}/setting_{}/dim_{}/".format(
        args.trans, args.mi_e, args.seed, args.setting, args.dim)
    isExist = os.path.exists(path)
    if not os.path.exists(path):
        os.makedirs(path)

    path = path + "/rho{}.json".format(args.rho)
    with open(path, 'w') as f:
        json.dump(r, f)
