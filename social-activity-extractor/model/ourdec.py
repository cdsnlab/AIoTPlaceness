import datetime
import os
from time import sleep

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
from model.util import align_cluster, count_percentage, pdist, do_tsne
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

class QCalculator(nn.Module):
    def __init__(self, n_clusters):
        super(self.__class__, self).__init__()
        self.n_clusters = n_clusters

        self.layer0 = nn.Sequential(
            nn.Linear(n_clusters*2, int(n_clusters*3/2)),
            nn.BatchNorm1d(int(n_clusters*3/2)),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Linear(int(n_clusters*3/2), n_clusters),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=1)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, image_q, text_q):
        output0 = self.layer0(torch.cat([image_q, text_q], dim=1))
        output1 = self.layer1(output0)
        q = self.softmax(output1)
        return q

# class FusionLayer(nn.Module):
#     def __init__(self, n_clusters):
#         super(self.__class__, self).__init__()
#         self.n_clusters = n_clusters
#
#         self.layer0 = nn.Sequential(
#             nn.Linear(n_clusters*2, n_clusters),
#             nn.ReLU()
#         )
#         self.layer1 = nn.Sequential(
#             nn.Linear(n_clusters, 2),
#             nn.Sigmoid()
#         )
#         self.softmax = nn.Softmax(dim=1)
#
#     def save_model(self, path):
#         torch.save(self.state_dict(), path)
#
#     def load_model(self, path):
#         pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
#         model_dict = self.state_dict()
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#         model_dict.update(pretrained_dict)
#         self.load_state_dict(model_dict)
#
#     def forward(self, p_image, p_text):
#         output0 = self.layer0(torch.cat([p_image, p_text], dim=1))
#         output1 = self.layer1(output0)
#         prob = self.softmax(output1)
#         return prob

class MultiDEC(nn.Module):
    def __init__(self, device, image_encoder, text_encoder, ours=False, use_prior=False, fl=False, n_clusters=10, alpha=1):
        super(self.__class__, self).__init__()
        self.device = device
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.ours = ours
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.use_prior = use_prior
        if use_prior:
            self.prior = torch.zeros(self.n_clusters).float().to(device)
        self.acc = 0.
        self.nmi = 0.
        self.f_1 = 0.
        self.softmax = nn.Softmax(dim=1)
        self.fl = fl
        if fl:
            self.weight_parameter = Parameter(torch.full((n_clusters,), 0.5))

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
        image_q = 1.0 / (1.0 + torch.sum((image_z.unsqueeze(1) - self.image_encoder.mu) ** 2, dim=2) / self.alpha)
        image_q = image_q ** (self.alpha + 1.0) / 2.0
        image_q = image_q / torch.sum(image_q, dim=1, keepdim=True)

        text_q = 1.0 / (1.0 + torch.sum((text_z.unsqueeze(1) - self.text_encoder.mu) ** 2, dim=2) / self.alpha)
        text_q = text_q ** (self.alpha + 1.0) / 2.0
        text_q = text_q / torch.sum(text_q, dim=1, keepdim=True)
        if self.fl:
            q = image_q * self.weight_parameter.expand_as(image_q) + text_q * (1 - self.weight_parameter).expand_as(text_q)
        else:
            q = torch.mean(torch.stack([image_q, text_q]), dim=0)
        return q

    def loss_function(self, p, q):
        h = torch.mean(p, dim=0, keepdim=True)

        if self.use_prior:
            unsupervised_loss = F.kl_div(q.log(), p, reduction='batchmean') + F.kl_div(self.prior.log(), h, reduction='batchmean')
        else:
            u = torch.full_like(h, fill_value=1/h.size()[1])
            unsupervised_loss = F.kl_div(q.log(), p, reduction='batchmean') + F.kl_div(u.log(), h, reduction='batchmean')
        return unsupervised_loss

    def semi_loss_function(self, label_batch, q_batch):
        supervised_loss = F.nll_loss(q_batch.log(), label_batch)
        return supervised_loss

    def target_distribution(self, q):
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit_predict(self, full_dataset, train_dataset, test_dataset, args, CONFIG, lr=0.001, batch_size=256, num_epochs=10, update_time=1, save_path=None, tol=1e-3, kappa=0.1):
        full_num = len(full_dataset)
        full_num_batch = int(math.ceil(1.0 * len(full_dataset) / batch_size))
        train_num = len(train_dataset)
        train_num_batch = int(math.ceil(1.0 * len(train_dataset) / batch_size))
        test_num = len(test_dataset)
        test_num_batch = int(math.ceil(1.0 * len(test_dataset) / batch_size))
        '''X: tensor data'''
        self.to(self.device)
        print("=====Training DEC=======")
        if args.adam:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)

        print("Extracting initial features at %s" % (str(datetime.datetime.now())))
        image_z = []
        text_z = []
        for batch_idx in range(full_num_batch):
            image_batch = full_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, full_num)][1]
            text_batch = full_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, full_num)][2]
            image_inputs = Variable(image_batch).to(self.device)
            text_inputs = Variable(text_batch).to(self.device)
            _image_z, _text_z = self.forward(image_inputs, text_inputs)
            image_z.append(_image_z.data.cpu())
            text_z.append(_text_z.data.cpu())
            del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z
        image_z = torch.cat(image_z, dim=0)
        text_z = torch.cat(text_z, dim=0)

        train_image_z = []
        train_text_z = []
        for batch_idx in range(train_num_batch):
            image_batch = train_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, train_num)][1]
            text_batch = train_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, train_num)][2]
            image_inputs = Variable(image_batch).to(self.device)
            text_inputs = Variable(text_batch).to(self.device)
            _image_z, _text_z = self.forward(image_inputs, text_inputs)
            train_image_z.append(_image_z.data.cpu())
            train_text_z.append(_text_z.data.cpu())
            del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z
        train_image_z = torch.cat(train_image_z, dim=0)
        train_text_z = torch.cat(train_text_z, dim=0)

        print("Initializing cluster centers with kmeans at %s" % (str(datetime.datetime.now())))
        image_kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        image_kmeans.fit(image_z.data.cpu().numpy())
        train_image_pred = image_kmeans.predict(train_image_z.data.cpu().numpy())
        print("Image kmeans completed at %s" % (str(datetime.datetime.now())))

        text_kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        text_kmeans.fit(text_z.data.cpu().numpy())
        train_text_pred = text_kmeans.predict(train_text_z.data.cpu().numpy())
        print("Text kmeans completed at %s" % (str(datetime.datetime.now())))

        full_short_codes = full_dataset[:][0]
        train_short_codes = train_dataset[:][0]
        test_short_codes = test_dataset[:][0]
        train_labels = train_dataset[:][3].squeeze(dim=0).data.cpu().numpy()
        test_labels = test_dataset[:][3].squeeze(dim=0).data.cpu().numpy()
        df_train = pd.DataFrame(data=train_labels, index=train_short_codes, columns=['label'])
        df_test = pd.DataFrame(data=test_labels, index=test_short_codes, columns=['label'])
        _, image_ind = align_cluster(train_labels, train_image_pred)
        _, text_ind = align_cluster(train_labels, train_text_pred)

        image_cluster_centers = np.zeros_like(image_kmeans.cluster_centers_)
        text_cluster_centers = np.zeros_like(text_kmeans.cluster_centers_)
        for i in range(self.n_clusters):
            image_cluster_centers[i] = image_kmeans.cluster_centers_[image_ind[i]]
            text_cluster_centers[i] = text_kmeans.cluster_centers_[text_ind[i]]
        self.image_encoder.mu.data.copy_(torch.Tensor(image_cluster_centers))
        self.text_encoder.mu.data.copy_(torch.Tensor(text_cluster_centers))
        if self.use_prior:
            for label in train_labels:
                self.prior[label] = self.prior[label] + 1
            self.prior /= len(train_labels)

        print("Calculating initial p at %s" % (str(datetime.datetime.now())))
        # update p considering short memory
        q = []
        for batch_idx in range(full_num_batch):
            image_batch = full_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, full_num)][1]
            text_batch = full_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, full_num)][2]

            image_inputs = Variable(image_batch).to(self.device)
            text_inputs = Variable(text_batch).to(self.device)

            _image_z, _text_z = self.forward(image_inputs, text_inputs)
            _q = self.soft_assignemt(_image_z, _text_z)
            q.append(_q.data.cpu())

            del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z, _q

        for batch_idx in range(test_num_batch):
            image_batch = test_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, test_num)][1]
            text_batch = test_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, test_num)][2]

            image_inputs = Variable(image_batch).to(self.device)
            text_inputs = Variable(text_batch).to(self.device)

            _image_z, _text_z = self.forward(image_inputs, text_inputs)
            _q = self.soft_assignemt(_image_z, _text_z)
            q.append(_q.data.cpu())

            del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z, _q

        q = torch.cat(q, dim=0)

        p = self.target_distribution(q)
        initial_pred = torch.argmax(p, dim=1).numpy()
        initial_acc = accuracy_score(test_labels, initial_pred[full_num:])
        initial_nmi = normalized_mutual_info_score(test_labels, initial_pred[full_num:], average_method='geometric')
        initial_f_1 = f1_score(test_labels, initial_pred[full_num:], average='macro')
        print("#Initial measure: acc: %.4f, nmi: %.4f, f_1: %.4f" % (initial_acc, initial_nmi, initial_f_1))
        df_initial = pd.DataFrame(data=initial_pred, index=full_short_codes + test_short_codes, columns=['label'])
        df_initial['pred'] = 'pred'
        df_initial.loc[df_train.index, 'pred'] = 'label'
        for idx, row in df_train.iterrows():
            df_initial.loc[idx, 'label'] = row['label']
        df_initial.loc[df_test.index, 'pred'] = 'label'
        for idx, row in df_test.iterrows():
            df_initial.loc[idx, 'label'] = row['label']

        if args.tsne:
            print("Conducting initial TSNE at %s" % (str(datetime.datetime.now())))
            do_tsne(p.numpy(), df_initial, self.n_clusters, os.path.join(CONFIG.SVG_PATH, args.gpu, 'epoch_000.png'))
            print("TSNE completed at %s" % (str(datetime.datetime.now())))

        flag_end_training = False
        for epoch in range(num_epochs):
            print("Epoch %d at %s" % (epoch, str(datetime.datetime.now())))
            # update the target distribution p
            self.train()
            # train 1 epoch
            train_unsupervised_loss = 0.0
            train_supervised_loss = 0.0
            adjust_learning_rate(lr, optimizer)
            for batch_idx in range(train_num_batch):
                # supervised phase
                image_batch = train_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, train_num)][1]
                text_batch = train_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, train_num)][2]
                label_batch = train_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, train_num)][3].squeeze(dim=0)

                optimizer.zero_grad()
                image_inputs = Variable(image_batch).to(self.device)
                text_inputs = Variable(text_batch).to(self.device)
                label_inputs = Variable(label_batch).to(self.device)

                _image_z, _text_z = self.forward(image_inputs, text_inputs)
                qbatch = self.soft_assignemt(_image_z, _text_z)
                supervised_loss = self.semi_loss_function(label_inputs, qbatch)
                train_supervised_loss += supervised_loss.data * len(label_inputs)
                supervised_loss.backward()
                optimizer.step()

                del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z

            # update p considering short memory
            q = []
            for batch_idx in range(full_num_batch):
                image_batch = full_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, full_num)][1]
                text_batch = full_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, full_num)][2]

                image_inputs = Variable(image_batch).to(self.device)
                text_inputs = Variable(text_batch).to(self.device)

                _image_z, _text_z = self.forward(image_inputs, text_inputs)
                _q = self.soft_assignemt(_image_z, _text_z)
                q.append(_q.data.cpu())

                del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z, _q
            q = torch.cat(q, dim=0)

            p = self.target_distribution(q)

            adjust_learning_rate(lr * kappa, optimizer)

            for batch_idx in range(full_num_batch):
                # clustering phase
                image_batch = full_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, full_num)][1]
                text_batch = full_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, full_num)][2]
                pbatch = p[batch_idx * batch_size: min((batch_idx + 1) * batch_size, full_num)]

                optimizer.zero_grad()
                image_inputs = Variable(image_batch).to(self.device)
                text_inputs = Variable(text_batch).to(self.device)
                p_inputs = Variable(pbatch).to(self.device)

                _image_z, _text_z = self.forward(image_inputs, text_inputs)
                qbatch = self.soft_assignemt(_image_z, _text_z)
                unsupervised_loss = self.loss_function(p_inputs, qbatch)
                train_unsupervised_loss += unsupervised_loss.data * len(p_inputs)
                unsupervised_loss.backward()
                optimizer.step()

                del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z
            train_unsupervised_loss /= full_num
            train_supervised_loss /= train_num

            train_pred = torch.argmax(q, dim=1).numpy()
            df_pred = pd.DataFrame(data=train_pred, index=full_short_codes, columns=['pred'])
            df_pred = df_pred.loc[df_train.index]
            train_pred = df_pred['pred']
            train_acc = accuracy_score(train_labels, train_pred)
            train_nmi = normalized_mutual_info_score(train_labels, train_pred, average_method='geometric')
            train_f_1 = f1_score(train_labels, train_pred, average='macro')
            print("#Train measure %3d: acc: %.4f, nmi: %.4f, f_1: %.4f" % (
                epoch + 1, train_acc, train_nmi, train_f_1))
            print("#Train loss %3d: unsup lss: %.4f, super lss: %.4f" % (
                epoch + 1, train_unsupervised_loss, train_supervised_loss))
            if epoch == 0:
                train_pred_last = train_pred
                train_unsupervised_loss_last = train_unsupervised_loss
            else:
                if args.es:
                    train_unsupervised_loss = train_unsupervised_loss
                    if train_unsupervised_loss_last > train_unsupervised_loss and epoch >= 5:
                        print("Reach local max/min loss. Stopping training.")
                        flag_end_training = True
                    train_unsupervised_loss_last = train_unsupervised_loss
                else:
                    delta_label = np.sum(train_pred != train_pred_last).astype(np.float32) / len(train_pred)
                    train_pred_last = train_pred
                    if delta_label < tol:
                        print('delta_label ', delta_label, '< tol ', tol)
                        print("Reach tolerance threshold. Stopping training.")
                        flag_end_training = True

            self.eval()
            test_unsupervised_loss = 0.0
            test_supervised_loss = 0.0
            # update p considering short memory
            q = []
            for batch_idx in range(full_num_batch):
                image_batch = full_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, full_num)][1]
                text_batch = full_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, full_num)][2]

                image_inputs = Variable(image_batch).to(self.device)
                text_inputs = Variable(text_batch).to(self.device)

                _image_z, _text_z = self.forward(image_inputs, text_inputs)
                _q = self.soft_assignemt(_image_z, _text_z)
                q.append(_q.data.cpu())

                del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z, _q

            for batch_idx in range(test_num_batch):
                image_batch = test_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, test_num)][1]
                text_batch = test_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, test_num)][2]
                label_batch = test_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, test_num)][3].squeeze(dim=0)

                image_inputs = Variable(image_batch).to(self.device)
                text_inputs = Variable(text_batch).to(self.device)
                label_inputs = Variable(label_batch).to(self.device)

                _image_z, _text_z = self.forward(image_inputs, text_inputs)
                qbatch = self.soft_assignemt(_image_z, _text_z)
                supervised_loss = self.semi_loss_function(label_inputs, qbatch)
                test_supervised_loss += supervised_loss.data * len(label_inputs)
                _q = self.soft_assignemt(_image_z, _text_z)
                q.append(_q.data.cpu())

                del image_batch, text_batch, label_batch, image_inputs, text_inputs, label_inputs, _image_z, _text_z, _q
            q = torch.cat(q, dim=0)
            test_p = self.target_distribution(q)

            if args.tsne and (epoch + 1) % 5 == 0:
                do_tsne(test_p.numpy(), df_initial, self.n_clusters, os.path.join(CONFIG.SVG_PATH, args.gpu, 'epoch_' + ('%03d' % (epoch + 1)) + '.png'))

            for batch_idx in range(full_num_batch):
                # clustering phase
                image_batch = full_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, full_num)][1]
                text_batch = full_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, full_num)][2]
                pbatch = test_p[batch_idx * batch_size: min((batch_idx + 1) * batch_size, full_num)]

                image_inputs = Variable(image_batch).to(self.device)
                text_inputs = Variable(text_batch).to(self.device)
                p_inputs = Variable(pbatch).to(self.device)

                _image_z, _text_z = self.forward(image_inputs, text_inputs)
                qbatch = self.soft_assignemt(_image_z, _text_z)
                unsupervised_loss = self.loss_function(p_inputs, qbatch)
                test_unsupervised_loss += unsupervised_loss.data * len(p_inputs)
                del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z
            test_unsupervised_loss /= full_num
            test_supervised_loss /= test_num

            test_pred = torch.argmax(q, dim=1).numpy()[full_num:]
            test_acc = accuracy_score(test_labels, test_pred)
            test_nmi = normalized_mutual_info_score(test_labels, test_pred, average_method='geometric')
            test_f_1 = f1_score(test_labels, test_pred, average='macro')
            print("#Test measure %3d: acc: %.4f, nmi: %.4f, f_1: %.4f" % (
                epoch + 1, test_acc, test_nmi, test_f_1))
            print("#Test loss %3d: unsup lss: %.4f, super lss: %.4f" % (
                epoch + 1, test_unsupervised_loss, test_supervised_loss))
            self.acc = test_acc
            self.nmi = test_nmi
            self.f_1 = test_f_1

            if flag_end_training:
                break

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
