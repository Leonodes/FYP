from __future__ import print_function
import sys
import os.path
import torch
import numpy as np
# import cupy as cp
import scipy.io as sio
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as D
import time
from torch.autograd import Variable


def make_dataset(root_path):
    path_ind = []

    for root, _, fnames in sorted(os.walk(root_path)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)  # path combination
            item = (path, int(os.path.splitext(fname)[0]))
            path_ind.append(item)
    # print(len(path_ind))
    return path_ind


def npy_loader(path):
    npy_file = np.load(path)
    Tensor_out = torch.from_numpy(npy_file.astype(float)).float()  # dimention to be checked
    # background = torch.from_numpy(mat_file['Background'].astype(float)).float()
    return Tensor_out  # (valset_x-background)/torch.max(valset_x-background)


def mat_loader(path):
    mat_file = sio.loadmat(path)
    valset_x = torch.from_numpy(mat_file['Speckles'].astype(float)).float()  # dimention to be checked
    # background = torch.from_numpy(mat_file['background'].astype(float)).float()
    return valset_x  # (valset_x-background)/torch.max(valset_x-background)


class SPKFolder(D.Dataset):

    def __init__(self, root, loader=npy_loader):
        sampling_speckle = make_dataset(root + '/speckle/')
        sampling_image = make_dataset(root + '/image/')
        if len(sampling_speckle) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.root = root
        self.loader = loader
        self.sampling_speckle = sampling_speckle
        self.sampling_image = sampling_image

    def __getitem__(self, index):
        # NSp_path, _ = self.sampling_Noisyspeckle[index]

        SP_path, _ = self.sampling_speckle[index]
        # Bin_path, _ = self.sampling_binning[index]

        # SP_path = NSp_path
        speckles = self.loader(SP_path)

        # Nspeckles = self.loader(NSp_path)
        # Binning = self.loader(Bin_path)

        Image_path, _ = self.sampling_image[index]
        image = self.loader(Image_path)

        return speckles.unsqueeze(0), image.unsqueeze(0)

    def __len__(self):
        return len(self.sampling_speckle)

if __name__ == '__main__':
    # ratio = '030'
    # root = 'D:/FFHQ/splitdata/16bit/ref_'+ ratio + '/medium1/test'
    root = 'C:/Users/Leon/Desktop/imageReconstruction/dataset'

    trainset = SPKFolder(root=root)

    print(trainset)

    # MatFolder(root=r'F:\movingdataset', num_image=num_image, batch_size=BATCH_S)
'''
if __name__ == '__main__':
    num_image = imageloading()
    BATCH_S = 1
    dataset = MatFolder(root=r'F:\movingdataset', num_image=num_image, batch_size=BATCH_S)
    dataloader = D.DataLoader(
        dataset=dataset,
        batch_size=BATCH_S,
        shuffle=True,
        num_workers=0,
    )
    print(len(dataloader))
    for batch_ind, (speckles, motion, image) in enumerate(dataloader):
        t_s = time.time()
        x = speckles
        y1 = motion
        y2 = image
        t = time.time()-t_s
        print(t)
'''
