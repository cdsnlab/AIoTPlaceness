import datetime

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
from model.util import align_cluster, count_percentage
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


class MDEC_encoder(nn.Module):
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


class MultiDEC(nn.Module):
    def __init__(self, device, image_encoder, text_encoder, n_clusters=10, alpha=1.):
        super(self.__class__, self).__init__()
        self.device = device
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.n_clusters = n_clusters
        self.alpha = alpha

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, image_x, text_x):
        image_z = self.image_encoder(image_x)
        text_z = self.text_encoder(text_x)
        return image_z, text_z

    def soft_assignemt(self, image_z, text_z):
        q = 1.0 / (1.0 + torch.sum((image_z.unsqueeze(1) - self.image_encoder.mu) ** 2, dim=2) / self.alpha)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)

        r = 1.0 / (1.0 + torch.sum((text_z.unsqueeze(1) - self.text_encoder.mu) ** 2, dim=2) / self.alpha)
        r = r ** (self.alpha + 1.0) / 2.0
        r = r / torch.sum(r, dim=1, keepdim=True)
        return q, r

    # def loss_function(self, p, q, r):
    #     def kld(target, pred):
    #         return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))
    #     h = torch.mean(p, dim=0, keepdim=True)
    #     u = torch.full_like(h, fill_value=1/h.size()[1])
    #     image_loss = kld(p, q) + kld(h, u)
    #     text_loss = kld(p, r) + kld(h, u)
    #     loss = image_loss + text_loss
    #     return loss

    def loss_function(self, p, q, r):
        h = torch.mean(p, dim=0, keepdim=True)
        u = torch.full_like(h, fill_value=1/h.size()[1])
        image_loss = F.kl_div(q.log(), p, reduction='batchmean') + F.kl_div(u.log(), h, reduction='batchmean')
        text_loss = F.kl_div(r.log(), p, reduction='batchmean') + F.kl_div(u.log(), h, reduction='batchmean')
        loss = image_loss + text_loss
        return loss

    def target_distribution(self, q, r):
        p_image = q ** 2 / torch.sum(q, dim=0)
        p_image = p_image / (2 * torch.sum(p_image, dim=1, keepdim=True))
        p_text = r ** 2 / torch.sum(r, dim=0)
        p_text = p_text / (2 * torch.sum(p_text, dim=1, keepdim=True))
        p = p_image + p_text
        return p

    def fit(self, X, lr=0.001, batch_size=256, num_epochs=10):
        num = len(X)
        num_batch = int(math.ceil(1.0 * len(X) / batch_size))
        '''X: tensor data'''
        self.to(self.device)
        print("=====Training DEC=======")
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)

        print("Extracting initial features at %s" % (str(datetime.datetime.now())))
        image_z = []
        text_z = []
        for batch_idx in range(num_batch):
            image_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][1]
            text_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][2]
            image_inputs = Variable(image_batch).to(self.device)
            text_inputs = Variable(text_batch).to(self.device)
            _image_z, _text_z = self.forward(image_inputs, text_inputs)
            image_z.append(_image_z.data.cpu())
            text_z.append(_text_z.data.cpu())
            del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z
            torch.cuda.empty_cache()
        image_z = torch.cat(image_z, dim=0)
        text_z = torch.cat(text_z, dim=0)

        print("Initializing cluster centers with kmeans at %s" % (str(datetime.datetime.now())))
        image_kmeans = KMeans(self.n_clusters, n_init=20)
        image_pred = image_kmeans.fit_predict(image_z.data.cpu().numpy())
        print("Image kmeans completed at %s" % (str(datetime.datetime.now())))

        text_kmeans = KMeans(self.n_clusters, n_init=20)
        text_pred = text_kmeans.fit_predict(text_z.data.cpu().numpy())
        print("Text kmeans completed at %s" % (str(datetime.datetime.now())))

        image_ind, text_ind = align_cluster(image_pred, text_pred)

        image_cluster_centers = np.zeros_like(image_kmeans.cluster_centers_)
        text_cluster_centers = np.zeros_like(text_kmeans.cluster_centers_)

        for i in range(self.n_clusters):
            image_cluster_centers[i] = image_kmeans.cluster_centers_[image_ind[i]]
            text_cluster_centers[i] = text_kmeans.cluster_centers_[text_ind[i]]
        self.image_encoder.mu.data.copy_(torch.Tensor(image_cluster_centers))
        self.image_encoder.mu.data = self.image_encoder.mu.cpu()
        self.text_encoder.mu.data.copy_(torch.Tensor(text_cluster_centers))
        self.text_encoder.mu.data = self.text_encoder.mu.cpu()
        self.train()
        for epoch in range(num_epochs):
            # update the target distribution p

            image_z = []
            text_z = []
            for batch_idx in range(num_batch):
                image_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][1]
                text_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][2]
                image_inputs = Variable(image_batch).to(self.device)
                text_inputs = Variable(text_batch).to(self.device)
                _image_z, _text_z = self.forward(image_inputs, text_inputs)
                image_z.append(_image_z.data.cpu())
                text_z.append(_text_z.data.cpu())
                del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z
                torch.cuda.empty_cache()
            image_z = torch.cat(image_z, dim=0)
            text_z = torch.cat(text_z, dim=0)

            q, r = self.soft_assignemt(image_z, text_z)
            p = self.target_distribution(q, r).data
            y_pred = torch.argmax(p, dim=1).numpy()
            count_percentage(y_pred)
            # train 1 epoch
            train_loss = 0.0
            for batch_idx in range(num_batch):
                image_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][1]
                text_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][2]
                pbatch = p[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]

                optimizer.zero_grad()
                image_inputs = Variable(image_batch).to(self.device)
                text_inputs = Variable(text_batch).to(self.device)
                target = Variable(pbatch)

                image_z, text_z = self.forward(image_inputs, text_inputs)
                qbatch, rbatch = self.soft_assignemt(image_z.cpu(), text_z.cpu())
                loss = self.loss_function(target, qbatch, rbatch)
                train_loss += loss.data * len(target)
                loss.backward()
                optimizer.step()

                del image_batch, text_batch, image_inputs, text_inputs, image_z, text_z
                torch.cuda.empty_cache()

            print("#Epoch %3d: Loss: %.4f at %s" % (
                epoch + 1, train_loss / num, str(datetime.datetime.now())))

    def fit_predict(self, X, batch_size=256):
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
            torch.cuda.empty_cache()
        short_codes = np.concatenate(short_codes, axis=0)
        image_z = torch.cat(image_z, dim=0)
        text_z = torch.cat(text_z, dim=0)

        q, r = self.soft_assignemt(image_z, text_z)
        p = self.target_distribution(q, r).data
        # y_pred = torch.argmax(p, dim=1).numpy()
        y_confidence, y_pred = torch.max(p, dim=1)
        y_confidence = y_confidence.numpy()
        y_pred = y_pred.numpy()
        count_percentage(y_pred)
        return short_codes, y_pred, y_confidence
