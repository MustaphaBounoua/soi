import numpy as np
import torch
from numpy.random import default_rng
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset, DataLoader
import itertools
from itertools import chain, combinations
from sklearn import preprocessing
from scipy.special import erf

N = 100000


SETTINGS_red = [[3, 3, 4], [3, 3], [3]]
SETTINGS_Syn = [[3, 3, 4], [3, 3], [3]]
SETTINGS_mix = [{"red": [3, 4], "syn": [3]}, {
    "red": [3], "syn": [3]}, {"red": [3], "syn": []}]


def get_dataloader(task, args):

    rescale = False if args.transformation == "" else True
    # If no transformation applied no need to rescale the data (already mean 0 and std 1).
    # Transformations concerves MI, O-information but not entropies.
    # If transformation is applied the entropies ground values truth is no longer valid.

    d_train, d_test = task.get_torch_dataset(
        args.N, args.N_test, dim=args.dim, rescale=rescale, seed=args.seed)

    train_loader, test_loader = DataLoader(d_train, batch_size=args.bs, shuffle=True, num_workers=args.nb_workers, drop_last=True, pin_memory=True), DataLoader(d_test, batch_size=1000, shuffle=False,
                                                                                                                                                                num_workers=args.nb_workers, drop_last=False, pin_memory=True)

    return train_loader, test_loader


def get_task(args, custom_setting=None):
    if custom_setting != None:
        args.benchmark = "custom"
        return Task_combination(tasks=[Task_redundant(nb_var=subsystem["nb"], rho=subsystem["rho"], dim=args.dim) if subsystem["type"] == "red"
                                       else Task_synergy(nb_var=subsystem["nb"], rho=subsystem["rho"], dim=args.dim)
                                       for subsystem in custom_setting],
                                dim=args.dim,
                                transformation=args.transformation)

    else:
        if args.benchmark == "red":
            setting = SETTINGS_red[args.setting]
            return Task_combination(tasks=[Task_redundant(nb_var=i, rho=args.rho, dim=args.dim)
                                           for i in setting], dim=args.dim, transformation=args.transformation)
        elif args.benchmark == "syn":
            setting = SETTINGS_Syn[args.setting]
            return Task_combination(tasks=[Task_synergy(nb_var=i, rho=args.rho, dim=args.dim)
                                           for i in setting], dim=args.dim, transformation=args.transformation)
        else:
            if hasattr(args, 'o_inf_order') and args.o_inf_order == 2:
                if args.setting == 0:
                    # in the paper : we use {
                    #     red : 3 var rho =0.7,
                    #     syn: 3 var rho =0.8,
                    #     red: 4 var rho =0.8
                    # }
                    return Task_combination(tasks=[Task_redundant(nb_var=3, rho=0.7 , dim=args.dim, #rho=args.rho
                                                                  ),
                                                   Task_synergy(
                                                       nb_var=3, rho=args.rho, dim=args.dim),
                                                   Task_redundant(nb_var=4, rho=args.rho, dim=args.dim),
                                                   ], dim=args.dim)
                elif args.setting == 1:
                    # in the paper : we use {
                    #     red : 3 var rho =0.6,
                    #     syn: 3 var rho =0.8,
                    # }
                    return Task_combination(tasks=[Task_redundant(nb_var=3,rho =0.6, #rho=args.rho,
                                                                  dim=args.dim),
                                                   Task_synergy(
                                                       nb_var=3, rho=args.rho, dim=args.dim),
                                                   ], dim=args.dim)
            else:
                setting = SETTINGS_mix[args.setting]
                tasks = []
                for i in setting["red"]:
                    tasks.append(Task_redundant(
                        nb_var=i, rho=args.rho, dim=args.dim))
                for i in setting["syn"]:
                    tasks.append(Task_synergy(nb_var=i, rho=0.7, dim=args.dim))
                return Task_combination(tasks=tasks, dim=args.dim, transformation=args.transformation)


def eye(ma, dim):
    """
    Constructs a block diagonal matrix with the given matrix repeated along the diagonal.
    """
    if isinstance(ma, (list, tuple, np.ndarray)):
        new_m = np.zeros((ma.shape[0], ma.shape[0], dim, dim))
        for i in range(ma.shape[0]):
            for j in range(ma.shape[0]):
                new_m[i][j] = ma[i][j] * np.eye(dim)
        return np.transpose(new_m, (0, 2, 1, 3)).reshape((ma.shape[0]*dim, ma.shape[0]*dim))
    else:
        return ma * np.eye(dim)


def entropy(cov, dim):
    """
    Calculates the entropy of a multivariate normal distribution with the given covariance matrix.

    Args:
        cov (np.ndarray): The covariance matrix.
        dim (int): The dimension of the distribution.

    Returns:
        float: The entropy of the distribution.
    """
    dist = multivariate_normal(mean=None, cov=cov)
    return dim * dist.entropy()


def get_cov_minus_i(cov, i):
    """
    Returns the covariance matrix with the specified variables removed.

    Args:
        cov (np.ndarray): The covariance matrix.
        i (list): The indices of the variables to be removed.

    Returns:
        np.ndarray: The modified covariance matrix.
    """
    i.sort()
    k = 0
    for j in i:
        j = j-k
        cov_list = cov.tolist()
        cov_list.pop(j)
        cov_list = np.array(cov_list).T.tolist()
        cov_list.pop(j)
        cov = np.array(cov_list).T
        k += 1
    return cov


def tc(cov, dim):
    """
    Calculates the total correlation of a multivariate normal distribution with the given covariance matrix.
    """
    nb_var = cov.shape[0]
    return np.sum([entropy(cov[i][i], dim) for i in range(nb_var)]) - entropy(cov, dim)


def o_inf(cov, dim):
    """
    Calculates the O-information of a multivariate normal distribution with the given covariance matrix.

    """
    # nb_var = cov.shape[0]
    # tc_i = [tc(get_cov_minus_i(cov, [i]), dim) for i in range(nb_var)]
    # return (2-nb_var) * tc(cov, dim) + np.sum(tc_i), tc_i
    return tc(cov, dim) - dtc(cov, dim)


def dtc(cov, dim):
    """
    Calculates the dual total correlation of a multivariate normal distribution with the given covariance matrix.
    """
    nb_var = cov.shape[0]
    return (nb_var-1) * tc(cov, dim) - np.sum(
        [tc(get_cov_minus_i(cov, [i]), dim)
         for i in range(cov.shape[0])
         ])


def s_inf(cov, dim):
    """
    Calculates the s-information of a multivariate normal distribution with the given covariance matrix.
    """
    nb_var = cov.shape[0]
    return (nb_var) * tc(cov, dim) - np.sum(
        [tc(get_cov_minus_i(cov, [i]), dim)
         for i in range(cov.shape[0])
         ])


def fill_cov(new_cov, index, cov):
    """
    Fills the given covariance matrix with the values from the specified submatrix.
    """
    for i in range(cov.shape[0]):
        for j in range(cov.shape[0]):
            new_cov[i+index][j+index] = cov[i][j]
    return new_cov


def combined_cov(covs):
    """
    Combines multiple covariance matrices into a single covariance matrix.
    """
    total_dim = np.sum([c.shape[0] for c in covs])
    new_cov = np.zeros((total_dim, total_dim))
    index = 0
    for i in range(len(covs)):
        new_cov = fill_cov(new_cov, index, covs[i])
        index += covs[i].shape[0]
    return new_cov


class Task():
    def __init__(self, nb_var, sigma, dim, transformation=None):
        """
        Initializes a Task object.

        Args:
            nb_var (int): The number of variables.
            sigma (float): Used to specify the intensity of the correlation.
            dim (int): The dimension of each variable.
            transformation (str): The type of transformation to be applied.
        """
        self.nb_var = nb_var
        self.sigma = sigma
        self.cov = None
        self.dim = dim
        self.transformation = transformation

    def get_summary(self):
        """
        Returns a summary of the task.

        Returns:
            dict: A dictionary containing various information measures and information about the task.
        """
        return {"tc": self.tc(),
                "dtc": self.dtc(),
                "o_inf": self.o_inf(),
                "s_inf": self.s_inf(),
                "tc_minus": [tc(get_cov_minus_i(self.cov, [i]), self.dim, )
                             for i in range(self.cov.shape[0])],
                "dtc_minus": [dtc(get_cov_minus_i(self.cov, [i]), self.dim)
                              for i in range(self.cov.shape[0])],
                "g_tc": self.grad_tc(),
                "g_dtc": self.grad_dtc(),
                "g_o_inf": self.grad_o_inf(),
                "g_s_inf": self.grad_s_inf(),
                "e_joint": entropy(self.cov, self.dim),
                "e_minus_i": [entropy(get_cov_minus_i(self.cov, [i]), self.dim)
                              for i in range(self.cov.shape[0])],
                "e_marg_i": [entropy(self.cov[i][i], self.dim)
                             for i in range(self.cov.shape[0])],
                "e_i_cond_slash": [
                    entropy(self.cov, self.dim) -
                    entropy(get_cov_minus_i(self.cov, [i]), self.dim)
                    for i in range(self.cov.shape[0])
        ],
            "e_cond_ij": [
                    entropy(get_cov_minus_i(self.cov, [
                            i+1]), self.dim) - entropy(get_cov_minus_i(self.cov, [0, i+1]), self.dim)
                    for i in range(self.cov.shape[0] - 1)
        ]
        }

    def tc(self):
        return tc(self.cov, self.dim)

    def dtc(self):
        return dtc(self.cov, self.dim)

    def o_inf(self):
        return o_inf(self.cov, self.dim)

    def s_inf(self):
        return s_inf(self.cov, self.dim)

    def grad_tc(self):
        tc_minus = [tc(get_cov_minus_i(self.cov, [i]),
                       self.dim)
                    for i in range(self.cov.shape[0])
                    ]
        return [self.tc() - tc for tc in tc_minus]

    def grad_dtc(self):
        dtc_minus = [dtc(get_cov_minus_i(self.cov, [i]),
                         self.dim)
                     for i in range(self.cov.shape[0])
                     ]
        return [self.dtc() - dtc for dtc in dtc_minus]

    def grad_o_inf(self):
        grad_tc = self.grad_tc()
        grad_dtc = self.grad_dtc()
        return [
            tc - dtc for tc, dtc in zip(grad_tc, grad_dtc)
        ]

    def grad_s_inf(self):
        grad_tc = self.grad_tc()
        grad_dtc = self.grad_dtc()
        return [
            tc + dtc for tc, dtc in zip(grad_tc, grad_dtc)
        ]

    def sample_cov(self, N, dim, seed):
        """
        Generates samples from the task's multivariate normal distribution.

        Args:
            N (int): The number of samples to generate.
            dim (int): The dimension of the distribution.
            seed (int): The seed for the random number generator.
        Returns:
            np.ndarray: The generated samples.
        """
        cov_d = eye(self.cov, dim)

        mu = np.zeros(cov_d.shape[0])
        samples = default_rng(seed=seed).multivariate_normal(
            mu, cov_d, size=(N))
        samples = samples.reshape(N, self.nb_var, dim)

        return samples

    def get_torch_dataset(self, N, T, dim=1, rescale=False, seed=42):
        """
        Generates a torch dataset from the task's samples.

        Args:
            N (int): The number of training samples.
            T (int): The number of test samples.
            dim (int): The dimension of the distribution.
            rescale (bool): Whether to rescale the data.
            seed (int): The seed for the random number generator.
        Returns:
            tuple: A tuple containing the training and test datasets.
        """
        data = self.sample_cov(N+T, dim=dim, seed=seed)

        data = data.reshape(N+T, self.nb_var * dim)

        if self.transformation == "H-C":
            data = data * np.sqrt(np.abs(data))
        elif self.transformation == "CDF":
            data = 0.5 * (1 + erf(data / 2**0.5))
        if rescale:
            data = preprocessing.scale(data)
        data = data.reshape(N+T, self.nb_var, dim)
        data_train = data[:N, :, :]
        data_test = data[N:, :, :]
        return SynthetitcDataset(data_train), SynthetitcDataset(data_test)


class Task_redundant(Task):
    def __init__(self, nb_var=3, sigma=0.01, dim=1, rho=None):
        """
        Initializes a Task_redundant object.

        Args:
            nb_var (int): The number of variables.
            sigma (float): The standard deviation.
            dim (int): The dimension of the distribution.
            normalized (bool): Whether to normalize the covariance matrix.
            rho (float): The correlation coefficient.
        """
        super().__init__(nb_var, sigma, dim)

        self.build_cov_pure_redundancy_sigma_normalized(rho=rho)

    def build_cov_pure_redundancy_sigma_normalized(self, rho):
        """
        Builds the covariance matrix for a pure redundancy task.
        Args:
            rho (float): The correlation coefficient.
        """
        if rho == None:
            sigma = self.sigma
            rho = 1/(1+sigma**2)

        cov = rho * np.ones((self.nb_var, self.nb_var))
        for i in range(self.nb_var):
            cov[i][i] = 1
        self.cov = cov


class Task_synergy(Task):
    def __init__(self, nb_var=3, sigma=0.1, dim=1, rho=None):
        """
        Initializes a Task_synergy object.

        Args:
            nb_var (int): The number of variables.
            sigma (float): The standard deviation.
            dim (int): The dimension of the distribution.
            rho (float): The correlation coefficient.
            normalized (bool): Whether to normalize the covariance matrix.
        """
        super().__init__(nb_var, sigma, dim=dim)
        self.rho = rho
        self.build_cov_pure_synergy_sigma_normalized()

    def build_cov_pure_synergy_sigma_normalized(self):
        """
        Builds the covariance matrix for a pure synergy task.
        """
        nb_syn = 1 + self.nb_var-2

        rho = self.rho * 1 / np.sqrt(nb_syn)

        cov = rho * np.zeros((self.nb_var, self.nb_var))

        cov[0][1] = 1 / np.sqrt(nb_syn)
        cov[1][0] = 1 / np.sqrt(nb_syn)

        for i in range(self.nb_var):
            cov[i][i] = 1
            if i > 1:
                cov[0][i] = 0
                cov[i][0] = 0
                cov[1][i] = rho
                cov[i][1] = rho
        self.cov = cov


class SynthetitcDataset(Dataset):

    def __init__(self, data):
        """
        Initializes a SynthetitcDataset object.
        Args:
            data (np.ndarray): The data for the dataset.
        """
        self.data = data
        self.nb_var = data.shape[1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {"x"+str(i): torch.tensor(self.data[idx][i]) for i in range(self.nb_var)}


class Task_combination(Task):
    def __init__(self, tasks=[], dim=1, transformation=None):
        """
        Initializes a Task_combination object.

        Args:
            tasks (list): A list of Task objects.
            dim (int): The dimension of the distribution.
            transformation (str): The type of transformation to be applied.
        """
        cov = combined_cov([task.cov for task in tasks])
        nb_var = cov.shape[0]
        self.tasks = tasks
        super().__init__(nb_var=nb_var, sigma=None, dim=dim, transformation=transformation)

        self.cov = cov
