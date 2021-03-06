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
from model.ops import BCELoss


class SingleClassifier(nn.Module):
    def __init__(self, device, input_dim=300, filter_num=64, n_classes=23):
        super(self.__class__, self).__init__()
        self.device = device
        self.input_dim = input_dim
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(1, filter_num, kernel_size=7, stride=2),
        #     nn.BatchNorm1d(filter_num),
        #     nn.ReLU()
        # )
        # self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(filter_num, filter_num*2, kernel_size=5, stride=2),
        #     nn.BatchNorm1d(filter_num*2),
        #     nn.ReLU()
        # )
        # self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(filter_num*2, filter_num*4, kernel_size=3, stride=2),
        #     nn.BatchNorm1d(filter_num*4),
        #     nn.ReLU()
        # )
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Sequential(
        #     nn.Linear(filter_num*4, n_classes),
        #     nn.Sigmoid()
        # )
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.BatchNorm1d(input_dim*2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(input_dim*2, input_dim*3),
            nn.BatchNorm1d(input_dim*3),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(input_dim*3, input_dim*4),
            nn.BatchNorm1d(input_dim*4),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(input_dim*4, n_classes),
            nn.Sigmoid()
        )
        self.softmax = nn.LogSoftmax(dim=1)
        self.score = 0.


    def forward(self, x):
        # x = x.unsqueeze(1)
        # out1 = self.maxpool1(self.conv1(x))
        # out2 = self.maxpool2(self.conv2(out1))
        # out3 = self.avgpool(self.conv3(out2))
        # h = self.fc(out3.squeeze())
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        h = self.fc(out3)
        log_prob = self.softmax(h)
        return log_prob

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def fit(self, train_dataset, val_dataset, input_modal, lr=0.001, batch_size=256, num_epochs=10, save_path=None):
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=True)
        validloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                  shuffle=False)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        criterion = nn.NLLLoss().to(self.device)
        self.to(self.device)

        best_acc = 0.0
        for epoch in range(num_epochs):
            self.train()
            # train 1 epoch
            train_loss = 0.0
            train_acc = 0
            for batch_idx, input_batch in enumerate(trainloader):
                feature_batch = Variable(input_batch[input_modal]).to(self.device)
                target_batch = Variable(input_batch[3]).to(self.device)
                optimizer.zero_grad()
                log_prob = self.forward(feature_batch)
                loss = criterion(log_prob, target_batch)
                loss.backward()
                optimizer.step()
                train_loss = train_loss + loss
                pred_batch = torch.argmax(log_prob, dim=1).cpu().numpy()
                train_acc = train_acc + sum(1 for x, y in zip(pred_batch, input_batch[3]) if x == y)
                del feature_batch, target_batch, log_prob, loss
            train_acc = train_acc / len(trainloader.dataset)

            # validate
            self.eval()
            valid_loss = 0.0
            valid_acc = 0
            for batch_idx, input_batch in enumerate(validloader):
                feature_batch = Variable(input_batch[input_modal]).to(self.device)
                target_batch = Variable(input_batch[3]).to(self.device)
                log_prob = self.forward(feature_batch)
                loss = criterion(log_prob, target_batch)
                valid_loss = valid_loss + loss
                pred_batch = torch.argmax(log_prob, dim=1).cpu().numpy()
                valid_acc = valid_acc + sum(1 for x, y in zip(pred_batch, input_batch[3]) if x == y)
                del feature_batch, target_batch, log_prob, loss

            valid_acc = valid_acc / len(validloader.dataset)
            print("#Epoch %3d: train acc: %.6f, valid acc: %.6f at %s" % (
                epoch + 1, train_acc, valid_acc,
                str(datetime.datetime.now())))
            print("#Epoch %3d: train loss: %.6f, valid loss: %.6f at %s" % (
                epoch + 1, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset), str(datetime.datetime.now())))

            if best_acc < valid_acc:
                best_acc = valid_acc
                self.save_model(save_path)
        self.score = best_acc
