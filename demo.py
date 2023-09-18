from io import BytesIO
from enum import IntEnum

import torch
import torch.nn as nn
import requests
import boto3
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


class Styles(IntEnum):

    BOHEMIAN = 0
    CLASSIC_AND_ANTIQUE = 1
    KOREAN_AND_ASIAN = 2
    LUXURY = 3
    MID_CENTURY_MODERN = 4
    MINIMAL = 5
    MODERN = 6
    NATURAL = 7
    SCANDINAVIAN = 8
    VINTAGE = 9



def get_render_meta():
    dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-2')

    table = dynamodb.Table('PROD_render_meta')
    response = table.scan()
    data = response['Items']

    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])

        if len(data) > 10000:
            break

    return data


def get_image(image_id):

    try:
        res = requests.get(f'https://resources.archisketch.com/images/{image_id}/550xAUTO/{image_id}.png', stream=True)
        image = Image.open(BytesIO(res.content)).convert('RGB')
        image = image.resize((256, 256))
        return image
    except:
        return None


model = torch.load('result/0411_1047/0_fold.pt', map_location='cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

compose = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

softmax = nn.Softmax(dim=1)

@torch.no_grad()
def predict(image):

    inp = compose(image)
    inp = inp.unsqueeze(0)
    inp = inp.to(device)
    out = model(inp)
    out = softmax(out).squeeze().detach().cpu().numpy()

    return out


def plot_style(style_predictions):

    plt.clf()
    fig = plt.figure(tight_layout=True)
    plt.bar(np.arange(10), style_predictions)
    plt.xticks(np.arange(10), [x.name for x in Styles], rotation=90)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    image = Image.open(buf)

    return image


render_meta = get_render_meta()


def process():


    meta = render_meta.pop(0)
    image_id = meta['id']
    image = get_image(image_id)
    style_predictions = meta['style_predictions']
    style_predictions = [float(style_predictions[x.name]) for x in Styles]
    prod_plot = plot_style(style_predictions)
    top1, top2 = np.argsort(style_predictions)[::-1][:2]
    prod_names = f'{Styles(top1).name}: {style_predictions[top1]:.4f}, {Styles(top2).name}: {style_predictions[top2]:.4f}'

    dev_style_predictions = predict(image)
    dev_plot = plot_style(dev_style_predictions)
    top1, top2 = np.argsort(dev_style_predictions)[::-1][:2]
    dev_names = f'{Styles(top1).name}: {dev_style_predictions[top1]:.4f}, {Styles(top2).name}: {dev_style_predictions[top2]:.4f}'
    
    return image, prod_plot, prod_names, dev_plot, dev_names



def app():    
    with gr.Blocks() as demo:
    
        with gr.Row():

            image = gr.Image(label='render image')

            with gr.Column():
                prod_style = gr.Image(label='현행 모델')
                prod_style_text = gr.Text(label='현행 모델 Top 2 Accuracy')

            with gr.Column():
                dev_style = gr.Image(label='신규 학습 모델')
                dev_style_text = gr.Text(label='신규 학습 모델 Top 2 Accuracy')
        
        with gr.Row():
            btn = gr.Button(label='next')

        btn.click(
            fn=process,
            outputs=[image, prod_style, prod_style_text, dev_style, dev_style_text]
        )

    demo.queue(concurrency_count=1)
    demo.launch(server_name='0.0.0.0', share=True)


if __name__ == '__main__':

    app()