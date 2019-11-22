import datetime
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import normalized_mutual_info_score, f1_score, accuracy_score
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

class ImageModel(nn.Module):
    def __init__(self, device, image_encoder):
        super(self.__class__, self).__init__()
        self.device = device
        self.image_encoder = image_encoder
        self.acc = 0.
        self.nmi = 0.
        self.f_1 = 0.

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        log_prob = self.image_encoder(x)
        return log_prob

    def fit(self, train_dataset, lr=0.001, batch_size=256, num_epochs=10, save_path=None, tol=1e-3):
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        criterion = nn.NLLLoss().to(self.device)
        self.to(self.device)

        for epoch in range(num_epochs):
            self.train()
            # train 1 epoch
            train_loss = 0.0
            train_pred = []
            train_labels = []
            for batch_idx, input_batch in enumerate(trainloader):
                feature_batch = Variable(input_batch[1]).to(self.device)
                target_batch = Variable(input_batch[2]).to(self.device)
                optimizer.zero_grad()
                log_prob = self.forward(feature_batch)
                loss = criterion(log_prob, target_batch)
                loss.backward()
                optimizer.step()
                train_loss = train_loss + loss.data
                pred_batch = torch.argmax(log_prob, dim=1).cpu().numpy()
                train_pred.extend(pred_batch)
                train_labels.extend(target_batch.cpu().numpy())
                del feature_batch, target_batch, log_prob, loss
            train_loss = train_loss / len(trainloader)
            train_acc = accuracy_score(train_labels, train_pred)
            train_nmi = normalized_mutual_info_score(train_labels, train_pred, average_method='geometric')
            train_f_1 = f1_score(train_labels, train_pred, average='macro', labels=np.unique(train_pred))
            print("#Epoch %3d: acc: %.4f, nmi: %.4f, f_1: %.4f, loss: %.4f at %s" % (
                epoch + 1, train_acc, train_nmi, train_f_1, train_loss, str(datetime.datetime.now())))
        if save_path:
            self.save_model(save_path)

    def predict(self, test_dataset, batch_size=256, use_de=False):
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False)
        self.to(self.device)
        self.eval()
        test_pred = []
        test_labels = []
        for batch_idx, input_batch in enumerate(testloader):
            feature_batch = Variable(input_batch[1]).to(self.device)
            target_batch = Variable(input_batch[2]).to(self.device)
            log_prob = self.forward(feature_batch)
            pred_batch = torch.argmax(log_prob, dim=1).cpu().numpy()
            test_pred.extend(pred_batch)
            test_labels.extend(target_batch.cpu().numpy())
            del feature_batch, target_batch, log_prob

        test_acc = accuracy_score(test_labels, test_pred)
        test_nmi = normalized_mutual_info_score(test_labels, test_pred, average_method='geometric')
        test_f_1 = f1_score(test_labels, test_pred, average='macro', labels=np.unique(test_pred))
        print("#Test acc: %.4f, Test nmi: %.4f, Test f_1: %.4f" % (
            test_acc, test_nmi, test_f_1))
        self.acc = test_acc
        self.nmi = test_nmi
        self.f_1 = test_f_1
