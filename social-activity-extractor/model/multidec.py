import datetime
import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import normalized_mutual_info_score
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
    def __init__(self, device, image_encoder, text_encoder, n_clusters=10, alpha=1, trade_off=1e-6):
        super(self.__class__, self).__init__()
        self.device = device
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.trade_off = trade_off
        self.score = 0.

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

    # def semi_loss_function(self, image_z_batch, text_z_batch, image_z, text_z, label_batch, train_labels):
    #     a = torch.zeros(len(label_batch), len(train_labels))
    #     for i in range(len(label_batch)):
    #         for j in range(len(train_labels)):
    #             if label_batch[i] != -1 and train_labels[j] != -1:
    #                 if label_batch[i] == train_labels[j]:
    #                     a[i][j] = 1
    #                 elif label_batch[i] != train_labels[j]:
    #                     a[i][j] = -1
    #
    #     image_loss = torch.sum(torch.sum(a * pdist(image_z_batch, image_z), dim=1))
    #     text_loss = torch.sum(torch.sum(a * pdist(text_z_batch, text_z), dim=1))
    #     semi_loss = image_loss + text_loss
    #     semi_loss = self.trade_off * semi_loss / len(label_batch)
    #     return semi_loss

    # def semi_loss_function(self, image_z, text_z, label):
    #     a = []
    #     for i in range(0, len(label) - 1):
    #         for j in range(i + 1, len(label)):
    #             if label[i] == -1 or label[j] == -1:
    #                 a.append(0.)
    #             elif label[i] == label[j]:
    #                 a.append(1.)
    #             else:
    #                 #a.append((0.))
    #                 a.append(-1.)
    #     a = torch.from_numpy(np.array(a, dtype=np.float32))
    #     image_dist = torch.pdist(image_z)
    #     image_loss = self.trade_off * torch.sum(a * image_dist) / len(label)
    #     text_dist = torch.pdist(text_z)
    #     text_loss = self.trade_off * torch.sum(a * text_dist) / len(label)
    #     semi_loss = image_loss + text_loss
    #     return semi_loss

    def semi_loss_function(self, label_batch, q_batch, r_batch):
        image_loss = F.nll_loss(q_batch.log(), label_batch)
        text_loss = F.nll_loss(r_batch.log(), label_batch)
        semi_loss = image_loss + text_loss
        return semi_loss


    def target_distribution(self, q, r):
        p_image = q ** 2 / torch.sum(q, dim=0)
        p_image = p_image / (2 * torch.sum(p_image, dim=1, keepdim=True))
        p_text = r ** 2 / torch.sum(r, dim=0)
        p_text = p_text / (2 * torch.sum(p_text, dim=1, keepdim=True))
        p = p_image + p_text
        return p

    def update_z(self, input, batch_size):
        input_num = len(input)
        input_num_batch = int(math.ceil(1.0 * len(input) / batch_size))
        image_z = []
        text_z = []
        for batch_idx in range(input_num_batch):
            image_batch = input[batch_idx * batch_size: min((batch_idx + 1) * batch_size, input_num)][1]
            text_batch = input[batch_idx * batch_size: min((batch_idx + 1) * batch_size, input_num)][2]
            image_inputs = Variable(image_batch).to(self.device)
            text_inputs = Variable(text_batch).to(self.device)
            _image_z, _text_z = self.forward(image_inputs, text_inputs)
            image_z.append(_image_z.data.cpu())
            text_z.append(_text_z.data.cpu())
            del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z
        # torch.cuda.empty_cache()
        image_z = torch.cat(image_z, dim=0)
        text_z = torch.cat(text_z, dim=0)

        q, r = self.soft_assignemt(image_z, text_z)
        p = self.target_distribution(q, r).data
        return image_z, text_z

    def fit_predict(self, X, train_dataset, test_dataset, lr=0.001, batch_size=256, num_epochs=10, update_time=1, save_path=None, tol=1e-3):
        X_num = len(X)
        X_num_batch = int(math.ceil(1.0 * len(X) / batch_size))
        train_num = len(train_dataset)
        train_num_batch = int(math.ceil(1.0 * len(train_dataset) / batch_size))
        '''X: tensor data'''
        self.to(self.device)
        self.image_encoder.mu.data = self.image_encoder.mu.cpu()
        self.text_encoder.mu.data = self.text_encoder.mu.cpu()
        print("=====Training DEC=======")
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=True)
        validloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)

        print("Extracting initial features at %s" % (str(datetime.datetime.now())))
        image_z, text_z = self.update_z(X, batch_size)
        train_image_z, train_text_z = self.update_z(train_dataset, batch_size)
        print("Initializing cluster centers with kmeans at %s" % (str(datetime.datetime.now())))
        image_kmeans = KMeans(self.n_clusters, n_init=20)
        image_kmeans.fit(image_z.data.cpu().numpy())
        train_image_pred = image_kmeans.predict(train_image_z.data.cpu().numpy())
        print("Image kmeans completed at %s" % (str(datetime.datetime.now())))

        text_kmeans = KMeans(self.n_clusters, n_init=20)
        text_kmeans.fit(text_z.data.cpu().numpy())
        train_text_pred = text_kmeans.predict(train_text_z.data.cpu().numpy())
        print("Text kmeans completed at %s" % (str(datetime.datetime.now())))

        short_codes = X[:][0]
        train_short_codes = train_dataset[:][0]
        train_labels = train_dataset[:][3].data.cpu().numpy()
        df_train = pd.DataFrame(data=train_labels, index=train_short_codes, columns=['label'])
        _, image_ind = align_cluster(train_labels, train_image_pred)
        _, text_ind = align_cluster(train_labels, train_text_pred)

        image_cluster_centers = np.zeros_like(image_kmeans.cluster_centers_)
        text_cluster_centers = np.zeros_like(text_kmeans.cluster_centers_)
        for i in range(self.n_clusters):
            image_cluster_centers[i] = image_kmeans.cluster_centers_[image_ind[i]]
            text_cluster_centers[i] = text_kmeans.cluster_centers_[text_ind[i]]
        self.image_encoder.mu.data.copy_(torch.Tensor(image_cluster_centers))
        self.image_encoder.mu.data = self.image_encoder.mu.cpu()
        self.text_encoder.mu.data.copy_(torch.Tensor(text_cluster_centers))
        self.text_encoder.mu.data = self.text_encoder.mu.cpu()

        for epoch in range(num_epochs):
            # update the target distribution p
            self.train()
            # train 1 epoch
            train_loss = 0.0

            adjust_learning_rate(lr, optimizer)

            for batch_idx in range(train_num_batch):
                # semi-supervised phase
                image_batch = train_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, train_num)][1]
                text_batch = train_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, train_num)][2]
                label_batch = train_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, train_num)][3]

                optimizer.zero_grad()
                image_inputs = Variable(image_batch).to(self.device)
                text_inputs = Variable(text_batch).to(self.device)
                label_inputs = Variable(label_batch)

                _image_z, _text_z = self.forward(image_inputs, text_inputs)
                qbatch, rbatch = self.soft_assignemt(_image_z.cpu(), _text_z.cpu())
                semi_loss = self.semi_loss_function(label_inputs, qbatch, rbatch)
                train_loss += semi_loss.data * len(label_inputs)
                semi_loss.backward()
                optimizer.step()

                del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z

            image_z, text_z = self.update_z(X, batch_size)
            q, r = self.soft_assignemt(image_z, text_z)
            p = self.target_distribution(q, r).data

            adjust_learning_rate(lr/10, optimizer)

            for batch_idx in range(X_num_batch):
                # clustering phase
                image_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, X_num)][1]
                text_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, X_num)][2]
                pbatch = p[batch_idx * batch_size: min((batch_idx + 1) * batch_size, X_num)]

                optimizer.zero_grad()
                image_inputs = Variable(image_batch).to(self.device)
                text_inputs = Variable(text_batch).to(self.device)
                p_inputs = Variable(pbatch)

                _image_z, _text_z = self.forward(image_inputs, text_inputs)
                qbatch, rbatch = self.soft_assignemt(_image_z.cpu(), _text_z.cpu())
                loss = self.loss_function(p_inputs, qbatch, rbatch)
                train_loss += loss.data * len(p_inputs)
                loss.backward()
                optimizer.step()

                del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z
            train_loss = train_loss / X_num

            train_pred = torch.argmax(p, dim=1).numpy()
            df_pred = pd.DataFrame(data=train_pred, index=short_codes, columns=['pred'])
            df_pred = df_pred.loc[df_train.index]
            train_pred = df_pred['pred']
            train_correct = sum(1 for x, y in zip(train_pred, train_labels) if x == y)
            train_acc = train_correct / len(train_pred)
            print("#Epoch %3d: Acc: %.4f where %d / %d, test nmi: %5f, loss: %.4f at %s" % (
                epoch + 1, train_acc, train_correct, len(train_pred), normalized_mutual_info_score(train_labels, train_pred, average_method='arithmetic'), train_loss, str(datetime.datetime.now())))
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
        test_labels = test_dataset[:][3]
        test_image_z, test_text_z = self.update_z(test_dataset, batch_size)
        image_z = torch.cat([image_z, test_image_z], dim=0)
        text_z = torch.cat([text_z, test_text_z], dim=0)

        q, r = self.soft_assignemt(image_z, text_z)
        test_p = self.target_distribution(q, r).data
        test_pred = torch.argmax(test_p, dim=1).numpy()[X_num:]
        test_correct = sum(1 for x, y in zip(test_pred, test_labels) if x == y)
        test_acc = test_correct / len(test_pred)
        print("test acc = %.4f where  %d / %d, test nmi: %5f" % (test_acc, test_correct, len(test_pred), normalized_mutual_info_score(test_labels, test_pred, average_method='arithmetic')))
        self.score = test_acc
        if save_path:
            self.save_model(os.path.join(save_path, "mdec_" + str(self.image_encoder.z_dim)) + '_' + str(
            self.n_clusters) + ".pt")

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
