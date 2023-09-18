import math
import random

import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split


def make_balanced_sampler(labels):

    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]

    return WeightedRandomSampler(weights, len(weights))


def calculate_class_imbalance_weight(labels):

    class_counts = np.bincount(labels)
    weights = len(labels) / class_counts

    return weights



def split_train_test(dataset, test_size=0.15):

    train_idx = []
    test_idx = []
    for x in np.unique(np.array(dataset.targets)):
        indicies = np.where(np.array(dataset.targets) == x)[0]
        random.shuffle(indicies)
        train_size = math.floor(len(indicies) * (1-test_size))
        train_idx.extend(indicies[:train_size].tolist())
        test_idx.extend(indicies[train_size:].tolist())

    train_idx.sort()
    test_idx.sort()

    # train_idx, test_idx = train_test_split(list(range(len(dataset))), train_size=0.85, stratify=dataset.targets)
    train_sampler = make_balanced_sampler(np.array(dataset.targets)[train_idx])
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    return train_dataset, test_dataset, train_sampler


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


def get_dataloader(root, opt):

    dataset = ImageFolder(root)
    train_dataset, test_dataset, train_sampler = split_train_test(dataset)

    crop = transforms.RandomResizedCrop(
        opt.image_size, scale=[0.8, 1.0], ratio=[0.9, 1.1]
    )
    rand_aug = transforms.RandAugment()

    random_crop = transforms.Lambda(lambda x: crop(x) if random.random() < 0.5 else x)
    random_aug = transforms.Lambda(lambda x: rand_aug(x) if random.random() < 0.5 else x)

    train_transform = transforms.Compose([
        random_crop,
        random_aug,
        transforms.Resize([opt.image_size, opt.image_size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                             std=[0.5, 0.5, 0.5])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize([opt.image_size, opt.image_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                             std=[0.5, 0.5, 0.5])
    ])

    train_collate_fn = CollateFn(train_transform)
    valid_collate_fn = CollateFn(valid_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        # shuffle=True,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        collate_fn=train_collate_fn,
        sampler=train_sampler,
        pin_memory=True
    )

    valid_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        collate_fn=valid_collate_fn,
        # sampler=test_sampler,
        pin_memory=True
    )
    
    return train_loader, valid_loader


if __name__ == '__main__':


    from easydict import EasyDict as edict

    opt = edict(
        model='resnext50d_32x4d',
        pretrained=True,
        num_classes=6,
        image_size=224,
        batch_size=128,
        num_workers=4,
        root='/data/RTC_Dataset'
    )

    train_loader, valid_loader = get_dataloader(opt.root, opt)