import os
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from dataset import CustomDataset
from classifier import Classifier, ClassifierV2
from utils import *
from trainer import Trainer


class CFG:

    seed = 5252
    epoch = 30
    n_splits = 5
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lr = 1e-4
    weight_decay = 1e-5
    batch_size = 32


def load_dataset(dataset_path='/data/style_classification'):

    X_train, y_train = [], []
    label_encoder = LabelEncoder()

    folder_list = sorted(glob.glob(os.path.join(dataset_path, '**')))
    for folder in folder_list:
        folder_name = os.path.basename(folder)
        images = [str(p) for p in Path(folder).glob("**/*") if p.suffix in {".png", ".jpg", ".jpeg"}]
        labels = [folder_name] * len(images)
        X_train.extend(images)
        y_train.extend(labels)

    y_train = label_encoder.fit_transform(y_train).tolist()
    return X_train, y_train, label_encoder


def load_model(num_classes, weight):

    model = Classifier(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weight)

    return model, optimizer, criterion


def train(model, optimizer, criterion, train_loader, valid_loader, fold, dir_path):

    best_score = 0
    trainer = Trainer(model, CFG.device, criterion, optimizer, fold, dir_path)
    early_stop = EarlyStop(patience=10)

    for epoch_index in range(CFG.epoch):

        trainer.train_epoch(train_loader, epoch_index)
        # trainer.valid_epoch(valid_loader, epoch_index)

        early_stop(trainer.train_mean_loss)

        if early_stop.early_stop:
            break

        if trainer.train_mean_f1score > best_score:
            best_score = trainer.train_mean_f1score
            torch.save(model, f'result/{dir_path}/{fold}_fold.pt')


def main():

    print(f'Pytorch version:[{torch.__version__}]')
    print(f"device:[{CFG.device}]")
    print(f"GPU : {torch.cuda.get_device_name(0)}")

    fix_seed(CFG.seed)

    X_train, y_train, label_encoder = load_dataset()

    # make directory
    new_directory_path = datetime.now().strftime('%m%d_%H%M')
    if not os.path.exists(new_directory_path):
        os.makedirs(f'result/{new_directory_path}', exist_ok=True)

    # kfold = StratifiedKFold(n_splits=CFG.n_splits)

    # for fold, (train_idx, valid_idx) in enumerate(kfold.split(X_train, y_train)):

        # train_dataset = CustomDataset(np.array(X_train)[train_idx], np.array(y_train)[train_idx], image_size=256, train=True)
        # valid_dataset = CustomDataset(np.array(X_train)[valid_idx], np.array(y_train)[valid_idx], image_size=256, train=False)

        # train_loader = DataLoader(dataset=train_dataset, batch_size=CFG.batch_size, shuffle=True)
        # valid_loader = DataLoader(dataset=valid_dataset, batch_size=CFG.batch_size, shuffle=True)

    train_dataset = CustomDataset(np.array(X_train), np.array(y_train), image_size=256, train=True)
    # valid_dataset = CustomDataset(np.array(X_train)[valid_idx], np.array(y_train)[valid_idx], image_size=256, train=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=CFG.batch_size, shuffle=True)
    # valid_loader = DataLoader(dataset=valid_dataset, batch_size=CFG.batch_size, shuffle=True)

    weight = np.bincount(np.array(y_train))
    weight = torch.tensor([1 - (x / len(y_train)) for x in weight], dtype=torch.float).to(CFG.device)
    model, optimizer, criterion = load_model(len(label_encoder.classes_), weight)
    model = model.to(CFG.device)
    train(model, optimizer, criterion, train_loader, None, 0, new_directory_path)
    torch.cuda.empty_cache()


if __name__ == '__main__':

    main()