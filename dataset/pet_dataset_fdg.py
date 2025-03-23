import os
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


class PetDataset(Dataset):
    def __init__(self, root_folder):

        self.fdg_files = sorted(os.listdir(os.path.join(root_folder, 'fdg_3D')))
        self.fdg_noise_files = sorted(os.listdir(os.path.join(root_folder, 'fdg_3D_noise')))
        self.k1_files = sorted(os.listdir(os.path.join(root_folder, 'k1')))
        self.k2_files = sorted(os.listdir(os.path.join(root_folder, 'k2')))
        self.k3_files = sorted(os.listdir(os.path.join(root_folder, 'k3')))
        self.k4_files = sorted(os.listdir(os.path.join(root_folder, 'k4')))
        self.ki_files = sorted(os.listdir(os.path.join(root_folder, 'ki')))
        self.vb_files = sorted(os.listdir(os.path.join(root_folder, 'vb')))
        self.root_folder = root_folder

    def __len__(self):
        return len(self.fdg_files)

    def __getitem__(self, idx):
        fdg_data = loadmat(os.path.join(self.root_folder, 'fdg_3D', self.fdg_files[idx]))['data']
        fdg_noise_data = loadmat(os.path.join(self.root_folder, 'fdg_3D_noise', self.fdg_noise_files[idx]))['data']
        k1_data = loadmat(os.path.join(self.root_folder, 'k1', self.k1_files[idx]))['data_new']
        k2_data = loadmat(os.path.join(self.root_folder, 'k2', self.k2_files[idx]))['data_new']
        k3_data = loadmat(os.path.join(self.root_folder, 'k3', self.k3_files[idx]))['data_new']
        k4_data = loadmat(os.path.join(self.root_folder, 'k4', self.k4_files[idx]))['data_new']
        ki_data = loadmat(os.path.join(self.root_folder, 'ki', self.ki_files[idx]))['data']
        vb_data = loadmat(os.path.join(self.root_folder, 'vb', self.vb_files[idx]))['data']

        k1_data_new = np.tile(k1_data[:, :, np.newaxis], 12)
        k2_data_new = np.tile(k2_data[:, :, np.newaxis], 12)
        k3_data_new = np.tile(k3_data[:, :, np.newaxis], 12)
        k4_data_new = np.tile(k4_data[:, :, np.newaxis], 12)
        # k4_data_new = k3_data_new

        return fdg_data, fdg_noise_data, k1_data_new, k2_data_new, k3_data_new, k4_data_new, ki_data, vb_data


