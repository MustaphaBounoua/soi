import json
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.libs.soi import SOI
from src.libs.util import get_samples
from src.vbn.vbn import VBNDataset
from experiments.config import get_config

parser = get_config()


parser.add_argument('--change', type=str, default="change", help ="options are change  or non_change stimuli")
parser.add_argument('--time_bin', type=int, default=0)


def benchmark_exp(args):

    
    r = {} 
    if args.setting == 0:
        structure = ['VISp', 'VISl', 'VISal']
    else:
        structure = ["VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm" ]
    
    
    if args.dim ==10:
        file_path= "/data/vbn_05/"
    elif args.dim == 25:
        file_path = "/data/vbn_002/"
    elif args.dim == 50:
        file_path = "/data/vbn_001/" 
       
    if args.arch == "mlp":
        total_dim = args.dim * len(structure)
        if total_dim <=30:
                hidden_dim = 128
        elif total_dim <=75:
                hidden_dim = 192
        elif total_dim <= 150:
                hidden_dim = 256
        else :
                hidden_dim = 128*3
         
        args.hidden_dim = hidden_dim
    else:
        args.hidden_dim = 120
        
    
    train_set = VBNDataset(structures=structure,
                           change = args.change, 
                           preprocess= True,
                           file_path= file_path,
                           time_bin= args.time_bin,
                           dim = args.dim,
                           )
    print(len(train_set))
    
    data_loader = DataLoader(train_set, batch_size=args.bs,shuffle=True, #pin_memory=True,
                                    num_workers=8, drop_last=True)

    device = "cuda" if args.accelerator == "gpu" else "cpu"
    
    test_samples = get_samples(
        data_loader, device=device, N=10000)
    
    model = SOI(args, nb_var=len(structure), var_list= {i: args.dim for i in structure})
 
    model.fit(data_loader, None)
    
    r = {"e": model.compute_o_inf(test_samples)}
    
    ## compute O_inf for each session/mouse
    sessions=train_set.get_sessions()
    r_s ={"o_inf":[],"s_inf":[],"tc":[],"dtc":[]}
    model.to(device)
    model.eval()
    for sess in sessions:
        out=model.compute_o_inf(sess)
        r_s["o_inf"].append(out["o_inf"])
        r_s["s_inf"].append(out["s_inf"])
        r_s["tc"].append(out["tc"])
        r_s["dtc"].append(out["dtc"])
    r["ses"]= r_s
    return r
        


if __name__ =="__main__":
    
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    
    r = benchmark_exp(args)
    
    
    path = "{}/soi_vbn/{}/{}/{}/seed_{}/setting_{}/dim_{}/time_bin_{}/".format(args.results_dir, args.arch,
                                                               args.benchmark, args.transformation,
                                                               args.seed, args.setting, args.dim,args.time_bin)
    
    isExist = os.path.exists(path)
    if not os.path.exists(path):
        os.makedirs(path)
        
    path = path + "/results_{}.json".format(args.change)
    with open(path, 'w') as f:
        json.dump(r, f)