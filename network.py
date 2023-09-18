import timm
import torch.nn as nn


class VanillaRegNet(nn.Module):

    def __init__(self, model='regnety_016', pretrained=True, num_classes=10):
        super().__init__()
        self.backbone = timm.create_model(model, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x
    

class VanillaRegNetV2(nn.Module):

    def __init__(self, model='regnety_016', pretrained=True, num_classes=6, emb_dim=256, num_heads=8):
        super().__init__()
        self.backbone = timm.create_model(model, pretrained=pretrained)
        self.bn = nn.BatchNorm1d(1000)
        self.query = nn.Linear(1000, emb_dim)
        self.key = nn.Linear(1000, emb_dim)
        self.value = nn.Linear(1000, emb_dim)
        self.mh_attn = nn.MultiheadAttention(emb_dim, num_heads)
        self.fc1 = nn.Linear(1000, emb_dim)
        self.fc2 = nn.Linear(emb_dim, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.bn(self.backbone(x))

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn, _ = self.mh_attn(q, k, v)
        x = self.fc1(x)
        x += attn
        x = self.relu(self.fc2(x))

        return x


class DenseNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        # feature extraction, out_dices -> layer
        # self.backbone = timm.create_model('densenet201', pretrained=True, feature_only=True, out_indices=(2, 4))

        self.backbone = timm.create_model('densenet201', pretrained=True)
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(1920*8*8)
        self.ln1 = nn.Linear(1920*8*8, 256)
        self.ln2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.backbone.forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.bn(self.dropout(x))
        x = self.relu(self.ln1(x))
        x = self.ln2(x)

        return x



if __name__ == '__main__':

    import torch

    m = DenseNet()
    inp = torch.rand(2, 3, 224, 224)
    o = m(inp)
    print(o)
    print(o.shape)