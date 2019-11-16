import datetime
import os

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import math
from model.Single_Classifier import SingleClassifier
from model.util import align_cluster, count_percentage
from sklearn.cluster import KMeans


class WeightCalculator(nn.Module):
    def __init__(self, device, input_dim=300):
        super(self.__class__, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim*2, input_dim * 3),
            nn.BatchNorm1d(input_dim * 3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim * 4),
            nn.BatchNorm1d(input_dim * 4),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(input_dim * 4, 2),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=1)


    def forward(self, image_x, text_x):
        x = torch.cat([image_x, text_x], dim=1)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        h = self.fc(out2)
        prob = self.softmax(h)
        return prob

    def loss_function(self, prob, weight_batch):
        # loss = torch.mean(torch.sum(-weight_batch * prob.log(), dim=1))
        loss = F.mse_loss(prob, weight_batch)
        return loss

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def fit_predict(self, train_dataset, val_dataset, lr=0.001, batch_size=256, num_epochs=10, save_path=None):
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=True)
        validloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                  shuffle=False)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        # criterion = nn.MSELoss().to(self.device)
        self.to(self.device)

        self.train()
        for epoch in range(num_epochs):
            # train 1 epoch
            train_loss = 0.0
            for batch_idx, input_batch in enumerate(trainloader):
                image_feature_batch = Variable(input_batch[1]).to(self.device)
                text_feature_batch = Variable(input_batch[2]).to(self.device)
                weight_batch = Variable(input_batch[4]).to(self.device)
                optimizer.zero_grad()
                prob = self.forward(image_feature_batch, text_feature_batch)
                loss = self.loss_function(prob, weight_batch)
                loss.backward()
                optimizer.step()
                train_loss = train_loss + loss
                del image_feature_batch, text_feature_batch, weight_batch, prob, loss
            print("#Epoch %3d: train loss: %.6f at %s" % (
                epoch + 1, train_loss / len(trainloader.dataset),
                str(datetime.datetime.now())))
        # validate
        self.eval()
        valid_loss = 0.0
        for batch_idx, input_batch in enumerate(validloader):
            image_feature_batch = Variable(input_batch[1]).to(self.device)
            text_feature_batch = Variable(input_batch[2]).to(self.device)
            weight_batch = Variable(input_batch[4]).to(self.device)
            prob = self.forward(image_feature_batch, text_feature_batch)
            loss = self.loss_function(prob, weight_batch)
            valid_loss = valid_loss + loss
            del image_feature_batch, text_feature_batch, weight_batch, prob, loss

        print("#valid loss: %.6f at %s" % (valid_loss / len(validloader.dataset),
            str(datetime.datetime.now())))
        self.save_model(save_path)