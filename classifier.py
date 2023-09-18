import timm
import torch
import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, num_classes=10):

        super(Classifier, self).__init__()

        self.model = timm.create_model('tf_efficientnet_b4', pretrained=True)
        self.dropout = nn.Dropout(p=0.5)
        self.ln1 = nn.Linear(1000, 32)
        self.ln2 = nn.Linear(32, num_classes)

    def forward(self, x):
        
        x = self.dropout(self.model(x))
        x = self.ln1(x)
        x = self.ln2(x)
        return x


class ClassifierV2(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.backbone = timm.create_model('tf_efficientnet_b4', features_only=True, out_indices=(2, 4), pretrained=True)

        self.flatten = nn.Flatten()
        
        # low level conv block
        self.low_conv_1 = nn.Conv2d(56, 28, 3)
        self.low_conv_2 = nn.Conv2d(28, 14, 3)
        self.low_bn = nn.BatchNorm1d(10976)
        self.low_linear = nn.Linear(10976, 1000)

        # high level conv
        self.high_conv_1 = nn.Conv2d(448, 224, 3)
        self.high_conv_2 = nn.Conv2d(224, 112, 3)
        self.high_bn = nn.BatchNorm1d(1792)
        self.high_linear = nn.Linear(1792, 1000)

        self.linear = nn.Linear(2000, 10)
    
    def forward(self, x):

        features = self.backbone(x)

        # low level feature
        low = self.low_linear(self.low_bn(self.flatten(self.low_conv_2(self.low_conv_1(features[0])))))

        # high level feature
        high = self.high_linear(self.high_bn(self.flatten(self.high_conv_2(self.high_conv_1(features[1])))))

        x = torch.cat((low, high), dim=-1)
        x = self.linear(x)

        return x