
import os

from config import Config as CONFIG

import sys
import csv
import random
import _pickle as cPickle

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.utils import save_image
import torchvision.models as models
from torch.autograd import Variable
from torchvision.datasets.folder import pil_loader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet152(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer

def process_dataset_images(src_path, dist_path):
    cudnn.benchmark = True
    if torch.cuda.is_available():
        device = torch.device('cuda')

    net = Net().to(device)
    net.eval()
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    df_data = pd.read_csv(src_path, encoding='utf-8')
    if not os.path.exists(dist_path):
        os.mkdir(dist_path)
    pbar = tqdm(total=df_data.shape[0])
    for index, row in df_data.iterrows():
        pbar.update(1)
        image_path = row['0']
        shortcode = row['shortcode']
        try:
            image = img_transform(pil_loader(image_path))
            torch.cuda.empty_cache()
            with torch.no_grad():
                image_data = torch.from_numpy(image).type(torch.FloatTensor).to(device)
            out = net(image_data)
            features = out.detach().cpu().numpy()
            with open(os.path.join(dist_path, shortcode + '.p'), 'wb') as f:
                cPickle.dump(features, f)
            del image_data, image, out, features
            f.close()
        except OSError as e:
            print(e)
            print(image_path)
    pbar.close()

def run(option):
    if option == 0:
        process_dataset_images(src_path=sys.argv[2], dist_path=sys.argv[3])
    else:
        print("This option does not exist!\n")


if __name__ == '__main__':
    run(int(sys.argv[1]))
