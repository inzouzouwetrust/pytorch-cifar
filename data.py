"""
Data Class
"""

###########
# Imports #
###########
import os
import pdb
import sys
import pickle
from collections import namedtuple

import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision import datasets


###########
# Globals #
###########
DataParams = namedtuple("DataParams", ["root", "train", 
                "transform", "target_transform", "download", "mini"])

DATA_ROOT_10 = "/scratch/artemis/azouaoui/datasets/CIFAR10/data"
DATA_ROOT_100 = "/scratch/artemis/azouaoui/datasets/CIFAR100/data"

DATA_ROOTS = {"cifar10": DATA_ROOT_10, "cifar100": DATA_ROOT_100}


#########
# Utils #
#########
def get_normalization_constants(dset_name="cifar10"):
    """
    Compute training mean and std per channel
    to normalize the data using the ``Normalize`` transform
    """
    assert dset_name in DATA_ROOTS.keys(), dset_name
    root = DATA_ROOTS[dset_name]

    train_params = DataParams(root=root,
                            train=True,
                            transform=None,
                            target_transform=None,
                            download=False,
                            mini=None)

    if dset_name == "cifar10":
        dset = CIFAR10(train_params)
    elif dset_name == "cifar100":
        dset = CIFAR100(train_params)
    data = dset.data

    means = np.mean(data, axis=(0, 1, 2)) / 255.
    stds = np.std(data, axis=(0, 1, 2)) / 255.

    return (means, stds)

###########
# Classes #
###########
class CIFAR10(datasets.cifar.CIFAR10):
    # Override parent class __init__ method
    def __init__(self, params : DataParams):
        self.params = params

        self.root = os.path.expanduser(params.root)
        self.transform = params.transform
        self.target_transform = params.target_transform
        self.train = params.train  # training set or test set

        if params.download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if params.mini is not None:
            mini = min(params.mini, len(self.data))
            self.data = self.data[:mini]
            self.targets = self.targets[:mini]
            assert len(self.data) == len(self.targets)
        self._load_meta()

class CIFAR100(datasets.cifar.CIFAR100):
    # Override parent class __init__ method
    def __init__(self, params : DataParams):
        self.params = params

        self.root = os.path.expanduser(params.root)
        self.transform = params.transform
        self.target_transform = params.target_transform
        self.train = params.train  # training set or test set

        if params.download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if params.mini is not None:
            mini = min(params.mini, len(self.data))
            self.data = self.data[:mini]
            self.targets = self.targets[:mini]
            assert len(self.data) == len(self.targets)
        self._load_meta()


########
# Main #
########
if __name__ == "__main__":

    pdb.set_trace()
    # Normalization constants
    means, stds = get_normalization_constants()
    print(f"Mean per channel: {means}")
    print(f"Std per channel: {stds}")

    # Dataset mini version
    pdb.set_trace()
    trfs = transforms.Compose([transforms.RandomHorizontalFlip(),
                               # randomly translate by 4 pixels in each direction
                               transforms.RandomAffine(degrees=0,
                                                       translate=(0.125, 0.125)),
                               transforms.ToTensor(),
                               transforms.Normalize(means, stds)])

    mini_params = DataParams(root=DATA_ROOT,
                        train=True,
                        transform=trfs,
                        target_transform=None,
                        download=False,
                        mini=100)

    mini_dset = CIFAR10(mini_params)

    assert len(mini_dset) == 100, len(mini_dset)

    mini_params_100 = DataParams(root=DATA_ROOT_100,
                                 train=True,
                                 transform=trfs,
                                 target_transform=None,
                                 download=False,
                                 mini=250)

    mini_dset_100 = CIFAR100(mini_params_100)

    assert len(mini_dset_100) == 250, len(mini_dset_100)

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset=mini_dset_100, batch_size=25, shuffle=True)

    for batch_idx, (data, target) in enumerate(loader):
        pdb.set_trace()
