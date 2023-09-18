import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


def get_transform(image_size):

    crop = transforms.RandomResizedCrop(image_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_aug = transforms.RandAugment()
    random_crop = transforms.Lambda(lambda x: crop(x) if random.random() < 0.5 else x)
    random_aug = transforms.Lambda(lambda x: rand_aug(x) if random.random() < 0.5 else x)

    train_transform = transforms.Compose([
        random_crop,
        random_aug,
        transforms.Resize([image_size, image_size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    valid_transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    return train_transform, valid_transform



class CollateFn:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):

        new_batch = {'image': [], 'label': []}
        for b in batch:
            x, y = b
            x = self.transform(x)
            new_batch['image'].append(x)
            new_batch['label'].append(y)
        del batch

        new_batch['image'] = torch.stack(new_batch['image'])
        new_batch['label'] = torch.LongTensor(new_batch['label'])

        return new_batch


class CustomDataset(Dataset):

    def __init__(self, X, y=None, image_size=256):

        self._X = X
        self._y = y
        self._image_size = image_size

    def __len__(self):
        return len(self._X)

    def __getitem__(self, item):

        image_path = self._X[item]
        if self._y is not None:
            label = self._y[item]

        image = Image.open(image_path).convert('RGB')

        return image, label