import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Tools import DataReading


class DataSet(torch.utils.data.Dataset):
    def __init__(self, task, dataPath, dataName, width, height, flag):
        super(DataSet, self).__init__()
        self.task = task
        self.path = dataPath
        self.name = dataName
        self.width = width
        self.height = height
        self.flag = flag
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        img = Image.open(self.path + self.task + '\\Images\\' + self.name[idx]).convert('L')
        img = np.asarray(img.resize((self.height, self.width), Image.NEAREST))
        mask = Image.open(self.path + self.task + '\\Masks\\' + self.name[idx])
        mask = np.asarray(mask.resize((self.height, self.width), Image.NEAREST))
        ###Image to Segmentation Map##############
        if self.task == 'CT':
            maskEso = np.where((mask[:, :, 0] == 255) &
                               (mask[:, :, 1] == 0) &
                               (mask[:, :, 2] == 0), 1, 0)
            maskGal = np.where((mask[:, :, 0] == 0) &
                               (mask[:, :, 1] == 255) &
                               (mask[:, :, 2] == 0), 1, 0)
            maskLiv = np.where((mask[:, :, 0] == 0) &
                               (mask[:, :, 1] == 0) &
                               (mask[:, :, 2] == 255), 1, 0)
            maskLKi = np.where((mask[:, :, 0] == 255) &
                               (mask[:, :, 1] == 255) &
                               (mask[:, :, 2] == 0), 1, 0)
            maskPan = np.where((mask[:, :, 0] == 255) &
                               (mask[:, :, 1] == 0) &
                               (mask[:, :, 2] == 255), 1, 0)
            maskSpl = np.where((mask[:, :, 0] == 0) &
                               (mask[:, :, 1] == 255) &
                               (mask[:, :, 2] == 255), 1, 0)
            maskSto = np.where((mask[:, :, 0] == 255) &
                               (mask[:, :, 1] == 255) &
                               (mask[:, :, 2] == 255), 1, 0)
            maskBac = np.where((mask[:, :, 0] == 0) &
                               (mask[:, :, 1] == 0) &
                               (mask[:, :, 2] == 0), 1, 0)
            mask = np.dstack((maskEso, maskGal, maskLiv, maskLKi, maskPan, maskSpl, maskSto, maskBac))
        elif self.task == 'Cell':
            maskCyt = np.where((mask == 255), 1, 0)
            maskNuc = np.where((mask == 128), 1, 0)
            maskBac = np.where((mask == 0), 1, 0)
            mask = np.dstack((maskCyt, maskNuc, maskBac))

        if self.flag is False:
            flagTemp = 0
        else:
            flagTemp = self.flag[idx]

        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask, flagTemp