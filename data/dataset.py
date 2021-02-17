import os
import torch.utils.data as data


class ImageDataset(data.Dataset):
    def __init__(self, path):
        self.x = []
        for dirpath, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                self.x.append(os.path.join(dirpath, fname))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return 1
