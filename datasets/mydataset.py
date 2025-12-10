import os
from PIL import Image
from torch.utils.data import Dataset
from utils import get_transform
import pandas as pd
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data_path, label_path, phase='train', resize=500):
        assert phase in ['train', 'val', 'test']
        df = pd.read_csv(label_path)
        self.label_path = np.array(df.values)
        self.data_path = data_path
        self.phase = phase
        self.resize = resize
        self.image_id = []
        self.num_classes = 8

        self.transform = get_transform(self.resize, self.phase)

    def __getitem__(self, item):
        filename = str(self.label_path[item][0])
        filename = filename.strip().replace("\r\n", "").replace("\n", "")
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            filename = filename + '.png'
        image_path = os.path.join(self.data_path, filename)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = int(self.label_path[item][1])
        return image, label

    def __len__(self):
        return len(self.label_path)
