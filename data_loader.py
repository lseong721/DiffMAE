import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
from tqdm import tqdm
from glob import glob
import argparse


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data):
        self.data = data
        self.len = len(self.data)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""

        return self.data[index]

    def __len__(self):
        return self.len

def normalize_vertex(v, return_statistics=False):
    v_mu = v.mean(0)
    v_std = (v - v_mu).std()
    v = (v - v_mu) / v_std
    if return_statistics:
        return v, v_mu, v_std
    else:
        return v

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []


    data_list = np.array(glob(os.path.join(args.data_dir, '*.pt')))
    # s = np.random.permutation(len(data_list))
    # data_list = data_list[s]
    data_max = torch.load('/hdd6/seongmin/DB/Hair/data_max.pt')
    data_min = torch.load('/hdd6/seongmin/DB/Hair/data_min.pt')

    train_ratio = 0.8
    n_train = int(len(data_list) * train_ratio)
    n_test = len(data_list) - n_train
    print('n_train: {0}, n_test: {1}'.format(n_train, n_test))
    # data_set = []
    for data_name in tqdm(data_list):
        key = data_name

        pt = torch.load(data_name)
        pt = pt.nan_to_num(0).permute(2, 0, 1).float()
        pt = (pt - data_min) / (data_max - data_min + 1e-8)

        data[key]["key"] = key
        data[key]["images"] = pt
        # data_set.append(pt)

    # data_set = torch.stack(data_set)
    # data_max = data_set.max(0)[0]
    # data_min = data_set.min(0)[0]
    # print(data_min.shape, data_max.shape)
    # torch.save(data_max, '/hdd6/seongmin/DB/Hair/data_max.pt')
    # torch.save(data_min, '/hdd6/seongmin/DB/Hair/data_min.pt')
    for idx, (k, v) in enumerate(data.items()):
        if idx < n_train:
            train_data.append(v)
        else:
            test_data.append(v)

    print('\n Data Partition: ', len(train_data))
    return train_data, test_data

def get_dataloaders(args):
    dataset = {}
    train_data, test_data = read_data(args)

    train_data = Dataset(train_data)
    test_data = Dataset(test_data)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.5) # ratio to be remained
    parser.add_argument('--total_epoch', type=int, default=10000)
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_name", type=str, default="diffmae")
    parser.add_argument("--data_name", type=str, default="butterfly")
    parser.add_argument("--data_dir", type=str, default="/hdd6/seongmin/DB/Hair/optim_neural_textures")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.multiprocessing.set_start_method('spawn')

    dataset = get_dataloaders(args)
    for i, batch in enumerate(dataset['train']):
        print(batch['V'].shape)