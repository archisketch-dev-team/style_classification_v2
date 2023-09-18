import glob

import tqdm
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

from classifier import Classifier


idx_to_class = {
    0: '가구수정',
    1: '걸레받이수정',
    2: '곰팡이',
    3: '꼬임',
    4: '녹오염',
    5: '들뜸',
    6: '면불량',
    7: '몰딩수정',
    8: '반점',
    9: '석고수정',
    10: '오염',
    11: '오타공',
    12: '울음',
    13: '이음부불량',
    14: '창틀,문틀수정',
    15: '터짐',
    16: '틈새과다',
    17: '피스',
    18: '훼손'
}


weight = torch.load('runs/dacon/valid_loss=1.14-valid_f1score=0.81.ckpt', map_location='cpu')
state_dict = weight['state_dict']
for key in list(state_dict.keys()):
    state_dict[key[6:]] = state_dict.pop(key)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Classifier(num_classes=19)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                             std=[0.5, 0.5, 0.5])
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
        preds.append(idx_to_class[output.argmax(dim=1).detach().cpu().numpy().item()])

df = pd.read_csv('/data/coco/sample_submission.csv')
df['label'] = preds
df.to_csv('/data/coco/baseline_submit.csv', index=False)
