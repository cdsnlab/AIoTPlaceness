import datetime
import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, f1_score
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import math
from model.util import align_cluster, count_percentage, pdist
from sklearn.cluster import KMeans


def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

def adjust_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

class UDEC_encoder(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_clusters=10,
                 encodeLayer=[400], activation="relu", dropout=0):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)

        self.n_clusters = n_clusters
        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)
        return z


class UniDEC(nn.Module):
    def __init__(self, device, encoder, use_prior=False, n_clusters=10, alpha=1):
        super(self.__class__, self).__init__()
        self.device = device
        self.encoder = encoder
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.use_prior = use_prior
        if use_prior:
            self.prior = torch.zeros(n_clusters).float()
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
        z = self.encoder(x)
        return z

    def soft_assignemt(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.encoder.mu) ** 2, dim=2) / self.alpha)
        q = q ** (self.alpha + 1.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def loss_function(self, p, q):
        h = torch.mean(p, dim=0, keepdim=True)

        if self.use_prior:
            loss = F.kl_div(q.log(), p, reduction='batchmean') + F.kl_div(self.prior.log(), h, reduction='batchmean')
        else:
            u = torch.full_like(h, fill_value=1/h.size()[1])
            loss = F.kl_div(q.log(), p, reduction='batchmean') + F.kl_div(u.log(), h, reduction='batchmean')
        return loss

    def semi_loss_function(self, label_batch, q_batch):
        semi_loss = F.nll_loss(q_batch.log(), label_batch)
        return semi_loss


    def target_distribution(self, q):
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / (torch.sum(p, dim=1, keepdim=True))
        return p

    def update_z(self, input, batch_size):
        input_num = len(input)
        input_num_batch = int(math.ceil(1.0 * len(input) / batch_size))
        z = []
        for batch_idx in range(input_num_batch):
            data_batch = input[batch_idx * batch_size: min((batch_idx + 1) * batch_size, input_num)][1]
            data_inputs = Variable(data_batch).to(self.device)
            _z  = self.forward(data_inputs)
            z.append(_z.data.cpu())
            del data_batch, data_inputs, _z
        # torch.cuda.empty_cache()
        z = torch.cat(z, dim=0)

        q = self.soft_assignemt(z)
        p = self.target_distribution(q).data
        return z

    def fit_predict(self, X, train_dataset, test_dataset, lr=0.001, batch_size=256, num_epochs=10, update_time=1, save_path=None, tol=1e-3, kappa=0.1):
        X_num = len(X)
        X_num_batch = int(math.ceil(1.0 * len(X) / batch_size))
        train_num = len(train_dataset)
        train_num_batch = int(math.ceil(1.0 * len(train_dataset) / batch_size))
        '''X: tensor data'''
        self.to(self.device)
        self.encoder.mu.data = self.encoder.mu.cpu()
        print("=====Training DEC=======")
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=True)
        validloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)

        print("Extracting initial features at %s" % (str(datetime.datetime.now())))
        z = self.update_z(X, batch_size)
        train_z = self.update_z(train_dataset, batch_size)
        print("Initializing cluster centers with kmeans at %s" % (str(datetime.datetime.now())))
        kmeans = KMeans(self.n_clusters, n_init=20)
        kmeans.fit(z.data.cpu().numpy())
        train_pred = kmeans.predict(train_z.data.cpu().numpy())
        print("kmeans completed at %s" % (str(datetime.datetime.now())))

        short_codes = X[:][0]
        train_short_codes = train_dataset[:][0]
        train_labels = train_dataset[:][2].data.cpu().numpy()
        df_train = pd.DataFrame(data=train_labels, index=train_short_codes, columns=['label'])
        _, ind = align_cluster(train_labels, train_pred)

        cluster_centers = np.zeros_like(kmeans.cluster_centers_)
        for i in range(self.n_clusters):
            cluster_centers[i] = kmeans.cluster_centers_[ind[i]]
        self.encoder.mu.data.copy_(torch.Tensor(cluster_centers))
        self.encoder.mu.data = self.encoder.mu.cpu()

        if self.use_prior:
            for label in train_labels:
                self.prior[label] = self.prior[label] + 1
            self.prior = self.prior / len(train_labels)
        for epoch in range(num_epochs):
            # update the target distribution p
            self.train()
            # train 1 epoch
            train_loss = 0.0
            semi_train_loss = 0.0

            adjust_learning_rate(lr, optimizer)

            for batch_idx in range(train_num_batch):
                # semi-supervised phase
                data_batch = train_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, train_num)][1]
                label_batch = train_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, train_num)][2]

                optimizer.zero_grad()
                data_inputs = Variable(data_batch).to(self.device)
                label_inputs = Variable(label_batch)

                _z = self.forward(data_inputs)
                qbatch = self.soft_assignemt(_z.cpu())
                semi_loss = self.semi_loss_function(label_inputs, qbatch)
                semi_train_loss += semi_loss.data * len(label_inputs)
                semi_loss.backward()
                optimizer.step()

                del data_batch, data_inputs, _z

            z = self.update_z(X, batch_size)
            q = self.soft_assignemt(z)
            p = self.target_distribution(q).data

            adjust_learning_rate(lr * kappa, optimizer)

            for batch_idx in range(X_num_batch):
                # clustering phase
                data_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, X_num)][1]
                pbatch = p[batch_idx * batch_size: min((batch_idx + 1) * batch_size, X_num)]

                optimizer.zero_grad()
                data_inputs = Variable(data_batch).to(self.device)
                p_inputs = Variable(pbatch)

                _z = self.forward(data_inputs)
                qbatch = self.soft_assignemt(_z.cpu())
                loss = self.loss_function(p_inputs, qbatch)
                train_loss += loss.data * len(p_inputs)
                loss.backward()
                optimizer.step()

                del data_batch, data_inputs, _z
            train_loss = train_loss / X_num
            semi_train_loss = semi_train_loss / train_num

            train_pred = torch.argmax(p, dim=1).numpy()
            df_pred = pd.DataFrame(data=train_pred, index=short_codes, columns=['pred'])
            df_pred = df_pred.loc[df_train.index]
            train_pred = df_pred['pred']
            train_acc = accuracy_score(train_labels, train_pred)
            train_nmi = normalized_mutual_info_score(train_labels, train_pred, average_method='geometric')
            train_f_1 = f1_score(train_labels, train_pred, average='macro')
            print("#Epoch %3d: acc: %.4f, nmi: %.4f, f_1: %.4f, loss: %.4f, semi_loss: %.4f, at %s" % (
                epoch + 1, train_acc, train_nmi, train_f_1, train_loss, semi_train_loss, str(datetime.datetime.now())))
            if epoch == 0:
                train_pred_last = train_pred
            else:
                delta_label = np.sum(train_pred != train_pred_last).astype(np.float32) / len(train_pred)
                train_pred_last = train_pred
                if delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break

        self.eval()
        test_labels = test_dataset[:][2].squeeze(dim=0)
        test_z = self.update_z(test_dataset, batch_size)
        z = torch.cat([z, test_z], dim=0)

        q = self.soft_assignemt(z)
        test_p = self.target_distribution(q).data
        test_pred = torch.argmax(test_p, dim=1).numpy()[X_num:]
        test_acc = accuracy_score(test_labels, test_pred)

        test_short_codes = test_dataset[:][0]
        test_short_codes = np.concatenate([short_codes, test_short_codes],axis=0)
        df_test = pd.DataFrame(data=torch.argmax(test_p, dim=1).numpy(), index=test_short_codes, columns=['labels'])
        df_test.to_csv('udec_label.csv', encoding='utf-8-sig')
        df_test_p = pd.DataFrame(data=test_p.data.numpy(), index=test_short_codes)
        df_test_p.to_csv('udec_p.csv', encoding='utf-8-sig')
        test_nmi = normalized_mutual_info_score(test_labels, test_pred, average_method='geometric')
        test_f_1 = f1_score(test_labels, test_pred, average='macro')
        print("#Test acc: %.4f, Test nmi: %.4f, Test f_1: %.4f" % (
            test_acc, test_nmi, test_f_1))
        self.acc = test_acc
        self.nmi = test_nmi
        self.f_1 = test_f_1
        if save_path:
            self.save_model(save_path)

    def predict(self, X, batch_size=256):
        num = len(X)
        num_batch = int(math.ceil(1.0 * len(X) / batch_size))
        self.to(self.device)
        self.image_encoder.mu.data = self.image_encoder.mu.cpu()
        self.text_encoder.mu.data = self.text_encoder.mu.cpu()

        self.eval()
        image_z = []
        text_z = []
        short_codes = []
        for batch_idx in range(num_batch):
            short_codes.append(X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][0])
            image_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][1]
            text_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][2]
            image_inputs = Variable(image_batch).to(self.device)
            text_inputs = Variable(text_batch).to(self.device)
            _image_z, _text_z = self.forward(image_inputs, text_inputs)
            image_z.append(_image_z.data.cpu())
            text_z.append(_text_z.data.cpu())
            del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z
        # torch.cuda.empty_cache()
        short_codes = np.concatenate(short_codes, axis=0)
        image_z = torch.cat(image_z, dim=0)
        text_z = torch.cat(text_z, dim=0)

        q, r = self.soft_assignemt(image_z, text_z)
        p = self.target_distribution(q, r).data
        # y_pred = torch.argmax(p, dim=1).numpy()
        y_confidence, y_pred = torch.max(p, dim=1)
        y_confidence = y_confidence.numpy()
        y_pred = y_pred.numpy()
        p = p.numpy()
        count_percentage(y_pred)
        return short_codes, y_pred, y_confidence, p
