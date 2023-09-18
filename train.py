import tqdm
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.utilities.seed import seed_everything
from easydict import EasyDict as edict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from PIL import Image

from lion import Lion
from classifier import Classifier
from dataset import CollateFn, CustomDataset, get_transform


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2



class CustomClassifier(L.LightningModule):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = Classifier(opt.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=opt.num_classes, average='micro')
        self.valid_acc = Accuracy(task='multiclass', num_classes=opt.num_classes, average='micro')
        self.train_f1 = F1Score(task='multiclass', num_classes=opt.num_classes, average='weighted')
        self.valid_f1 = F1Score(task='multiclass', num_classes=opt.num_classes, average='weighted')
        self.train_prec = Precision(task='multiclass', num_classes=opt.num_classes, average='weighted')
        self.valid_prec = Precision(task='multiclass', num_classes=opt.num_classes, average='weighted')
        self.train_recall = Recall(task='multiclass', num_classes=opt.num_classes, average='weighted')
        self.valid_recall = Recall(task='multiclass', num_classes=opt.num_classes, average='weighted')
        self.train_cm = ConfusionMatrix(task='multiclass', num_classes=opt.num_classes)
        self.valid_cm = ConfusionMatrix(task='multiclass', num_classes=opt.num_classes)
   
    def training_step(self, batch, batch_idx):
        image, label = batch['image'], batch['label']
        # if np.random.rand(1) < 0.5:  # cut-mix
        #     lam = np.random.beta(0.5, 0.5)
        #     rand_index = torch.randperm(image.size()[0]).to(self.device)
        #     target_a = label
        #     target_b = label[rand_index]
        #     bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
        #     image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
        #     # adjust lambda to exactly match pixel ratio
        #     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
        #     # compute output
        #     y_pred = self.model(image)
        #     loss = self.criterion(y_pred, target_a) * lam + self.criterion(y_pred, target_b) * (1. - lam)
        # else:
        y_pred = self.model(image)
        loss = self.criterion(y_pred, label)

        self.train_acc(y_pred, label)
        self.train_f1(y_pred, label)
        self.train_prec(y_pred, label)
        self.train_recall(y_pred, label)
        self.train_cm(y_pred, label)

        self.log_dict({
            'train_loss': loss,
            'train_acc': self.train_acc,
            'train_f1score': self.train_f1,
            'train_precision': self.train_prec,
            'train_recall': self.train_recall
        }, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch['image'], batch['label']
        y_pred = self.model(image)

        return {'y_pred': y_pred, 'label': label}
    
    def validation_epoch_end(self, outputs):
        y_pred = torch.vstack([x['y_pred'] for x in outputs])
        label = torch.hstack([x['label'] for x in outputs])
        cm = self.valid_cm(y_pred, label)
        fig = plt.figure()
        df = pd.DataFrame(cm.cpu().numpy(), index=range(19), columns=range(19))
        sns.heatmap(df/np.sum(df), annot=True, cmap='Blues', vmin=0, vmax=1, fmt='.2f', annot_kws={"size": 7})
        self.trainer.logger.experiment.add_figure('valid_confusion_matrix', fig, global_step=self.global_step)
        
        loss = self.criterion(y_pred, label)
        self.valid_acc(y_pred, label)
        self.valid_f1(y_pred, label)
        self.valid_prec(y_pred, label)
        self.valid_recall(y_pred, label)
        
        self.log_dict({
            'valid_loss': loss,
            'valid_acc': self.valid_acc,
            'valid_f1score': self.valid_f1,
            'valid_precision': self.valid_prec,
            'valid_recall': self.valid_recall,
        }, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = Lion(self.parameters(), lr=1e-4, weight_decay=1e-5)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.99)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer, 
        #     base_lr=5e-6, 
        #     max_lr=3e-4, 
        #     step_size_up=100, 
        #     step_size_down=200, 
        #     cycle_momentum=False
        # )
        return optimizer
        # return {
        # "optimizer": optimizer,
        # "lr_scheduler": {
            # "scheduler": scheduler,
            # "interval": "step",
            # },
        # }


if __name__ == '__main__':

    opt = edict(
        num_classes=19,
        image_size=256,
        batch_size=32,
        num_workers=4,
        n_splits=5
    )

    seed_everything(5252)
    logger = TensorBoardLogger(save_dir='logs')

    images = glob.glob('/data/coco/train/**/*.png')
    labels = [x.split('/')[4] for x in images]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # images = np.array(images)
    # labels = np.array(labels)

    # kfold = StratifiedKFold(n_splits=opt.n_splits)
    # for fold, (train_idx, valid_idx) in enumerate(kfold.split(images, labels)):


    #     # train
    #     train_images = images[train_idx]
    #     train_labels = labels[train_idx]

    #     # valid
    #     valid_images = images[valid_idx]
    #     valid_labels = labels[valid_idx]

    #     train_dataset = CustomDataset(train_images, train_labels)
    #     valid_dataset = CustomDataset(valid_images, valid_labels)

    #     train_transform, valid_transform = get_transform(opt.image_size)

    #     train_collate_fn = CollateFn(train_transform)
    #     valid_collate_fn = CollateFn(valid_transform)

    #     train_loader = DataLoader(train_dataset, 
    #                               shuffle=True, 
    #                               batch_size=opt.batch_size,
    #                               num_workers=opt.num_workers,
    #                               collate_fn=train_collate_fn,
    #                               pin_memory=True)
    #     valid_loader = DataLoader(valid_dataset,
    #                               shuffle=False,
    #                               batch_size=opt.batch_size,
    #                               num_workers=opt.num_workers,
    #                               collate_fn=valid_collate_fn,
    #                               pin_memory=True)
        
    #     checkpoint_callbacks = ModelCheckpoint(
    #         dirpath=f'runs/dacon/{fold}', 
    #         filename='checkpoint',
    #         monitor='valid_loss',
    #         save_last=False,
    #         save_top_k=1,
    #         save_weights_only=True,
    #         mode='min',
    #         every_n_epochs=1
    #     )

    #     earlystop_callbacks = EarlyStopping(
    #         monitor='valid_loss',
    #         patience=4
    #     )

    #     classifier = CustomClassifier(opt=opt)

    #     trainer = Trainer(
    #         accumulate_grad_batches=16,
    #         logger=logger,
    #         callbacks=[checkpoint_callbacks, earlystop_callbacks],
    #         max_epochs=30,
    #         check_val_every_n_epoch=1,
    #         accelerator='gpu',
    #         precision=16,
    #         log_every_n_steps=10
    #     )
    #     trainer.fit(classifier, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    

    total_preds = []
    for fold in range(opt.n_splits):

        weight = torch.load(f'runs/dacon/{fold}/checkpoint.ckpt', map_location='cpu')
        state_dict = weight['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key[6:]] = state_dict.pop(key)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Classifier(num_classes=opt.num_classes)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        transform = transforms.Compose([
            transforms.Resize([opt.image_size, opt.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        images = sorted(glob.glob('/data/coco/test/*.png'), key=lambda x: int(x.split('/')[-1].split('.')[0]))
        preds = []
        with torch.no_grad():
            for image in tqdm.tqdm(images):
                image = Image.open(image)
                image = transform(image)
                image = image.unsqueeze(0)
                image = image.to(device)
                output = model(image)
                preds.append(output.detach().cpu().numpy().tolist())
        
        total_preds.append(preds)

    total_preds = np.array(total_preds).squeeze()
    total_preds = np.sum(total_preds, axis=0)
    total_preds = np.argmax(total_preds, axis=1)

    total_preds = label_encoder.inverse_transform(total_preds)

    df = pd.read_csv('/data/coco/sample_submission.csv')
    df['label'] = total_preds
    df.to_csv('/data/coco/baseline_submit.csv', index=False)