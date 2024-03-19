
import torch


def get_samples(test_loader, mod_list):
    data = {mod: torch.Tensor() for mod in mod_list}
    for batch in test_loader:
            for mod in mod_list:
                data[mod] = torch.cat([data[mod], batch[mod]])
    return data