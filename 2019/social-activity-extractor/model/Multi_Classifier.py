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


class MultiClassifier(nn.Module):
    def __init__(self, device, image_classifier, text_classifier, weight_calculator=None, fixed_weight=None):
        super(self.__class__, self).__init__()
        self.device = device
        self.image_classifier = image_classifier
        self.text_classifier = text_classifier
        self.weight_calculator = weight_calculator
        self.fixed_weight = fixed_weight
        self.softmax = nn.LogSoftmax(dim=1)
        self.score = 0.

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, image_x, text_x, weight):
        image_h = self.image_classifier(image_x)
        text_h = self.text_classifier(text_x)
        if self.weight_calculator is None and self.fixed_weight is not None:
            # weight = weight.unsqueeze(dim=-1)
            # h = torch.stack([image_h, text_h], dim=2)
            # h = torch.bmm(h, weight).squeeze()
            h = self.fixed_weight * image_h + (1 - self.fixed_weight) * text_h
        else:
            weight = self.weight_calculator(image_x, text_x).unsqueeze(dim=-1)
            h = torch.stack([image_h, text_h], dim=2)
            h = torch.bmm(h, weight).squeeze()
        log_prob = self.softmax(h)
        return log_prob

    def fit(self, train_dataset, val_dataset, lr=0.001, batch_size=256, num_epochs=10, save_path=None):
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
                image_feature_batch = Variable(input_batch[1]).to(self.device)
                text_feature_batch = Variable(input_batch[2]).to(self.device)
                target_batch = Variable(input_batch[3]).to(self.device)
                weight_batch = Variable(input_batch[4]).to(self.device)
                optimizer.zero_grad()
                log_prob = self.forward(image_feature_batch, text_feature_batch, weight_batch)
                loss = criterion(log_prob, target_batch)
                loss.backward()
                optimizer.step()
                train_loss = train_loss + loss
                pred_batch = torch.argmax(log_prob, dim=1).cpu().numpy()
                train_acc = train_acc + sum(1 for x, y in zip(pred_batch, input_batch[3]) if x == y)
                del image_feature_batch, text_feature_batch, target_batch, log_prob, loss
            train_acc = train_acc / len(trainloader.dataset)

            # validate
            self.eval()
            valid_loss = 0.0
            valid_acc = 0
            for batch_idx, input_batch in enumerate(validloader):
                image_feature_batch = Variable(input_batch[1]).to(self.device)
                text_feature_batch = Variable(input_batch[2]).to(self.device)
                target_batch = Variable(input_batch[3]).to(self.device)
                weight_batch = Variable(input_batch[4]).to(self.device)
                log_prob = self.forward(image_feature_batch, text_feature_batch, weight_batch)
                loss = criterion(log_prob, target_batch)
                valid_loss = valid_loss + loss
                pred_batch = torch.argmax(log_prob, dim=1).cpu().numpy()
                valid_acc = valid_acc + sum(1 for x, y in zip(pred_batch, input_batch[3]) if x == y)
                del image_feature_batch, text_feature_batch, target_batch, log_prob, loss

            valid_acc = valid_acc / len(validloader.dataset)
            print("#Epoch %3d: train acc: %.6f, valid acc: %.6f at %s" % (
                epoch + 1, train_acc, valid_acc,
                str(datetime.datetime.now())))
            print("#Epoch %3d: train loss: %.6f, valid loss: %.6f at %s" % (
                epoch + 1, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset),
                str(datetime.datetime.now())))

            if best_acc < valid_acc:
                best_acc = valid_acc
                self.save_model(save_path)
        self.score = best_acc

