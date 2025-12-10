from .mydataset import MyDataset

import os

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_data_path = os.path.join(_base_dir, "")
train_label_path = os.path.join(_base_dir, "")
test_data_path = os.path.join(_base_dir, "")
test_label_path = os.path.join(_base_dir, "")


def get_trainval_datasets(tag, resize):
    if tag == "CT":
        return MyDataset(train_data_path, train_label_path, 'train', resize=resize), MyDataset(test_data_path, test_label_path, 'val', resize=resize)
    else:
        raise ValueError('Unsupported Tag {}. Only "CT" tag is supported.'.format(tag))