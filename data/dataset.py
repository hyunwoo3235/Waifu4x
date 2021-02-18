import os
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


class ImageDataset(data.Dataset):
    def __init__(self, path):
        self.transform = transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.5, p=0.2),
            transforms.RandomRotation((-30, 30)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomChoice([
                transforms.RandomResizedCrop((256, 256), scale=(0.6, 1.0)),
                transforms.RandomCrop((256, 256), pad_if_needed=True, padding_mode='edge'),
                transforms.RandomCrop((256, 256), pad_if_needed=True, padding_mode='edge'),
            ]),
            transforms.ToTensor()
        ])
        self.resize = transforms.Compose([
            transforms.RandomChoice([
                transforms.Resize(0.25, interpolation=Image.NEAREST),
                transforms.Resize(0.25, interpolation=Image.BILINEAR),
                transforms.Resize(0.25, interpolation=Image.BICUBIC),
            ])
        ])
        self.x = []
        for dirpath, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                self.x.append(os.path.join(dirpath, fname))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        img = Image.open(self.x[index])
        GT = self.transform(img)
        LR = self.resize(GT)
        return {'GT': GT, 'LR': LR}
