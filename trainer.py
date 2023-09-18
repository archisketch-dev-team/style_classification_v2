import os

import tqdm
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score

from utils import rand_bbox


class Trainer:

    def __init__(self, model, device, criterion, optimizer, fold, dir_path):

        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.fold = fold
        self.dir_path = dir_path

    def train_epoch(self, train_loader, epoch_index):

        self.model.train()
        self.train_loss = 0

        orig_y_preds = []
        y_preds = []
        y_trues = []

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)

        for step, (image, label) in pbar:
            self.optimizer.zero_grad()

            image = image.to(self.device)
            label = label.to(self.device)

            # cut mix
            # if np.random.random() > 0.5:
            #     lam = np.random.beta(1., 1.)
            #     rand_index = torch.randperm(image.size()[0]).to(self.device)
            #     target_a = label
            #     target_b = label[rand_index]
            #     bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
            #     image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
            #     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
            #     output = self.model(image)
            #     loss = self.criterion(output, target_a) * lam + self.criterion(output, target_b) * (1. - lam)
            # else:
            output = self.model(image)
            loss = self.criterion(output, label)

            self.train_loss += loss
            y_pred = np.argmax(output.data.cpu().numpy(), axis=1)
            y_preds.extend(y_pred.tolist())
            orig_y_preds.extend(output.data.cpu().numpy())
            y_trues.extend(label.cpu().numpy().tolist())

            description = f'loss: {loss.item():.4f}'
            pbar.set_description(description)
            
            loss = loss / 4
            loss.backward()
            if (step + 1) % 4 == 0 or step + 1 == len(train_loader):
                self.optimizer.step()

        self.train_mean_loss = self.train_loss / len(train_loader)
        self.train_mean_acc = np.mean(np.array(y_preds) == np.array(y_trues))
        self.train_top2_acc = top_k_accuracy_score(np.array(y_trues), np.array(orig_y_preds))
        self.train_mean_precision = precision_score(np.array(y_trues), np.array(y_preds), average='macro')
        self.train_mean_recall = recall_score(np.array(y_trues), np.array(y_preds), average='macro')
        self.train_mean_f1score = f1_score(np.array(y_trues), np.array(y_preds), average='macro')
        df = pd.DataFrame(confusion_matrix(np.array(y_trues), np.array(y_preds)), index=range(10), columns=range(10))
        sns.heatmap(df/np.sum(df), annot=True, cmap='Blues', vmin=0, vmax=1, fmt='.2f', annot_kws={"size": 7})
        msg = f'Fold {self.fold} Epoch {epoch_index}, Train, loss: {self.train_mean_loss:.4f}, acc: {self.train_mean_acc:.4f}, top 2 acc: {self.train_top2_acc:.4f}, prec: {self.train_mean_precision:.4f} recall: {self.train_mean_recall:.4f}, f1 score: {self.train_mean_f1score:.4f}'
        print(msg)
        if not os.path.exists(f'result/{self.dir_path}/{self.fold}'):
            os.makedirs(f'result/{self.dir_path}/{self.fold}')
        plt.savefig(f'result/{self.dir_path}/{self.fold}/train_{epoch_index}_cm.png')
        plt.clf()

    def valid_epoch(self, valid_loader, epoch_index):

        self.model.eval()
        self.valid_loss = 0

        orig_y_preds = []
        y_preds = []
        y_trues = []

        pbar = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader), position=0, leave=True)

        with torch.no_grad():

            for step, (image, label) in pbar:

                image = image.to(self.device)
                label = label.to(self.device)

                output = self.model(image)
                loss = self.criterion(output, label)
                self.valid_loss += loss

                y_pred = np.argmax(output.data.cpu().numpy(), axis=1)
                y_preds.extend(y_pred.tolist())
                orig_y_preds.extend(output.data.cpu().numpy())
                y_trues.extend(label.cpu().numpy().tolist())

        self.valid_mean_loss = self.valid_loss / len(valid_loader)
        self.valid_mean_acc = np.mean(np.array(y_preds) == np.array(y_trues))
        self.valid_top2_acc = top_k_accuracy_score(np.array(y_trues), np.array(orig_y_preds))
        self.valid_mean_precision = precision_score(np.array(y_trues), np.array(y_preds), average='macro')
        self.valid_mean_recall = recall_score(np.array(y_trues), np.array(y_preds), average='macro')
        self.valid_mean_f1score = f1_score(np.array(y_trues), np.array(y_preds), average='macro')
        df = pd.DataFrame(confusion_matrix(np.array(y_trues), np.array(y_preds)), index=range(10), columns=range(10))
        sns.heatmap(df/np.sum(df), annot=True, cmap='Blues', vmin=0, vmax=1, fmt='.2f', annot_kws={"size": 7})
        msg = f'Fold {self.fold} Epoch {epoch_index}, Valid, loss: {self.valid_mean_loss:.4f}, acc: {self.valid_mean_acc:.4f}, top 2 acc: {self.valid_top2_acc:.4f}, prec: {self.valid_mean_precision:.4f} recall: {self.valid_mean_recall:.4f}, f1 score: {self.valid_mean_f1score:.4f}'
        print(msg)
        if not os.path.exists(f'result/{self.dir_path}/{self.fold}'):
            os.makedirs(f'result/{self.dir_path}/{self.fold}')
        plt.savefig(f'result/{self.dir_path}/{self.fold}/valid_{epoch_index}_cm.png')
        plt.clf()