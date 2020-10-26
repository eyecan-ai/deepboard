from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np


class BoardDataset(Dataset):

    def __init__(self, path):
        self.images = sorted(Path(path).glob('*.exr'))
        self.labels = sorted(Path(path).glob('*.txt'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]), -1)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image = np.transpose(image, (2, 0, 1))
        label = np.loadtxt(self.labels[idx], ndmin=1)

        result = {
            'image': image.astype(np.float32),
            'ratio': label.astype(np.float32)
        }

        return result
