import collections
import datetime
import os

import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

import config
from model.submodules import TextProcessor, Attention, Classifier
from sklearn.metrics import normalized_mutual_info_score, f1_score, accuracy_score

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import math

CONFIG = config.Config
class DualNet(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, device, pretrained_embedding, text_features, z_dim, n_classes):
        super(DualNet, self).__init__()
        self.device = device
        vision_features = CONFIG.OUTPUT_FEATURES
        glimpses = 2

        self.text = TextProcessor(
            pretrained_embedding=pretrained_embedding,
            lstm_features=text_features,
            drop=0.5,
        )
        self.attention = Attention(
            v_features=vision_features,
            q_features=text_features,
            mid_features=512,
            glimpses=2,
            drop=0.5,
        )

        self.fc = nn.Sequential(
            nn.Linear(glimpses * vision_features + text_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, z_dim),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, n_classes),
            nn.LogSoftmax(dim=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len):
        q = self.text(q, list(q_len.data))

        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        a = self.attention(v, q)
        v = apply_attention(v, a)

        combined = torch.cat([v, q], dim=1)
        z = self.fc(combined)
        return z

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def classify(self, z):
        log_prob = self.classifier(z)
        return log_prob

    def fit(self, train_dataset, test_dataset, lr=0.001, batch_size=128, num_epochs=10, save_path=None):
        trainloader = DataLoader(train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn)
        testloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        criterion = nn.NLLLoss().to(self.device)
        self.to(self.device)
        for epoch in range(num_epochs):
            # train 1 epoch
            self.train()
            train_loss = 0.0
            train_pred = []
            train_labels = []
            for batch_idx, input_batch in enumerate(trainloader):
                image_batch = Variable(input_batch[1]).to(self.device)
                text_batch = Variable(input_batch[2]).to(self.device)
                text_len_batch = Variable(input_batch[3]).to(self.device)
                target_batch = Variable(input_batch[4]).to(self.device)
                optimizer.zero_grad()
                z = self.forward(image_batch, text_batch, text_len_batch)
                log_prob = self.classify(z)
                loss = criterion(log_prob, target_batch)
                loss.backward()
                optimizer.step()
                train_loss = train_loss + loss.data
                pred_batch = torch.argmax(log_prob, dim=1).cpu().numpy()
                train_pred.extend(pred_batch)
                train_labels.extend(target_batch.cpu().numpy())
                del image_batch, text_batch, text_len_batch, target_batch, log_prob, loss
            train_loss = train_loss / len(trainloader)
            train_acc = accuracy_score(train_labels, train_pred)
            train_nmi = normalized_mutual_info_score(train_labels, train_pred, average_method='geometric')
            train_f_1 = f1_score(train_labels, train_pred, average='macro', labels=np.unique(train_pred))
            print("#Train Epoch %3d: acc: %.4f, nmi: %.4f, f_1: %.4f, loss: %.4f at %s" % (
                epoch + 1, train_acc, train_nmi, train_f_1, train_loss, str(datetime.datetime.now())))

            self.eval()
            test_loss = 0.0
            test_pred = []
            test_labels = []
            for batch_idx, input_batch in enumerate(testloader):
                image_batch = Variable(input_batch[1]).to(self.device)
                text_batch = Variable(input_batch[2]).to(self.device)
                text_len_batch = Variable(input_batch[3]).to(self.device)
                target_batch = Variable(input_batch[4]).to(self.device)
                z = self.forward(image_batch, text_batch, text_len_batch)
                log_prob = self.classify(z)
                loss = criterion(log_prob, target_batch)
                test_loss = test_loss + loss.data
                pred_batch = torch.argmax(log_prob, dim=1).cpu().numpy()
                test_pred.extend(pred_batch)
                test_labels.extend(target_batch.cpu().numpy())
                del image_batch, text_batch, text_len_batch, target_batch, log_prob, loss
            test_loss = test_loss / len(testloader)
            test_acc = accuracy_score(test_labels, test_pred)
            test_nmi = normalized_mutual_info_score(test_labels, test_pred, average_method='geometric')
            test_f_1 = f1_score(test_labels, test_pred, average='macro', labels=np.unique(test_pred))
            print("#Test Epoch %3d: acc: %.4f, nmi: %.4f, f_1: %.4f, loss: %.4f at %s" % (
                epoch + 1, test_acc, test_nmi, test_f_1, test_loss, str(datetime.datetime.now())))
        if save_path:
            self.save_model(save_path)



class DDEC(nn.Module):
    def __init__(self, device, pretrained_model, n_classes, z_dim, use_prior=False, alpha=1.):
        super(self.__class__, self).__init__()
        self.device = device
        self.dualnet = pretrained_model
        self.mu = Parameter(torch.Tensor(n_classes, z_dim))
        self.n_classes = n_classes
        self.z_dim = z_dim
        self.use_prior = use_prior
        if use_prior:
            self.prior = torch.zeros(n_classes).float()
        self.alpha = alpha
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

    def forward(self, v, q, q_len):
        z = self.dualnet(v, q, q_len)
        return z

    def soft_assignemt(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def target_distribution(self, q):
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def loss_function(self, p, q):
        h = torch.mean(p, dim=0, keepdim=True)
        u = torch.full_like(h, fill_value=1 / h.size()[1])
        loss = F.kl_div(q.log(), p, reduction='batchmean') + F.kl_div(u.log(), h, reduction='batchmean')
        return loss

    def semi_loss_function(self, label_batch, q_batch):
        semi_loss = F.nll_loss(q_batch.log(), label_batch)
        return semi_loss

    # def update_z(self, loader):
    #     z = []
    #     for batch_idx, input_batch in enumerate(loader):
    #         image_batch = Variable(input_batch[1]).to(self.device)
    #         text_batch = Variable(input_batch[2]).to(self.device)
    #         text_len_batch = Variable(input_batch[3]).to(self.device)
    #         _z = self.forward(image_batch, text_batch, text_len_batch)
    #         z.append(_z)
    #         del image_batch, text_batch, text_len_batch, _z
    #     z = torch.cat(z, dim=0)
    #     return z

    # def update_z(self, input, batch_size):
    #     input_num = len(input)
    #     input_num_batch = int(math.ceil(1.0 * len(input) / batch_size))
    #     z = []
    #     for batch_idx in range(input_num_batch):
    #         image_batch = input[batch_idx * batch_size: min((batch_idx + 1) * batch_size, input_num)][1]
    #         text_batch = input[batch_idx * batch_size: min((batch_idx + 1) * batch_size, input_num)][2]
    #         text_len_batch = input[batch_idx * batch_size: min((batch_idx + 1) * batch_size, input_num)][3]
    #         image_inputs = Variable(image_batch).to(self.device)
    #         text_inputs = Variable(text_batch).to(self.device)
    #         text_len_inputs = Variable(text_len_batch).to(self.device)
    #         _z = self.forward(image_inputs, text_inputs, text_len_inputs)
    #         z.append(_z.data.cpu())
    #         del image_batch, text_batch, image_inputs, text_inputs, text_len_inputs, _z
    #     z = torch.cat(z, dim=0)
    #     return z

    def fit(self, full_dataset, train_dataset, test_dataset, lr=0.001, batch_size=256, num_epochs=10, update_time=1, save_path=None, tol=1e-3, kappa=0.1):
        full_num = len(full_dataset)
        full_num_batch = int(math.ceil(1.0 * len(full_dataset) / batch_size))
        train_num = len(train_dataset)
        train_num_batch = int(math.ceil(1.0 * len(train_dataset) / batch_size))
        '''X: tensor data'''
        print("Training at %s" % (str(datetime.datetime.now())))
        self.to(self.device)
        self.dualnet = nn.DataParallel(self.dualnet)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)

        full_loader = DataLoader(full_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=CONFIG.DATA_WORKERS,
                                 collate_fn=collate_fn)
        train_loader = DataLoader(train_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=CONFIG.DATA_WORKERS,
                                 collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=CONFIG.DATA_WORKERS,
                                 collate_fn=collate_fn)
        #z = []
        short_codes = []
        for batch_idx, input_batch in enumerate(tqdm(full_loader, desc="Extracting short codes", total=len(full_loader))):
            short_codes.extend(list(input_batch[0]))
            # image_batch = Variable(input_batch[1]).to(self.device)
            # text_batch = Variable(input_batch[2]).to(self.device)
            # text_len_batch = Variable(input_batch[3]).to(self.device)
            # _z = self.forward(image_batch, text_batch, text_len_batch)
            # z.append(_z.data.cpu())
            # del image_batch, text_batch, text_len_batch, _z
        #z = torch.cat(z, dim=0)

        train_z = []
        train_short_codes = []
        train_labels = []
        for batch_idx, input_batch in enumerate(tqdm(train_loader, desc="Extracting initial cluster means", total=len(train_loader))):
            train_short_codes.extend(list(input_batch[0]))
            image_batch = Variable(input_batch[1]).to(self.device)
            text_batch = Variable(input_batch[2]).to(self.device)
            text_len_batch = Variable(input_batch[3]).to(self.device)
            train_labels.extend(input_batch[4].tolist())
            _z = self.forward(image_batch, text_batch, text_len_batch)
            train_z.append(_z.data.cpu())
            del image_batch, text_batch, text_len_batch, _z
        train_z = torch.cat(train_z, dim=0)
        df_train = pd.DataFrame(data=train_labels, index=train_short_codes, columns=['label'])

        cluster_means = torch.zeros((self.n_classes, self.z_dim))
        num_clusters = torch.zeros(self.n_classes)
        for i, label in enumerate(train_labels):
            cluster_means[label] = cluster_means[label] + train_z[i]
            num_clusters[label] = num_clusters[label] + 1
        cluster_means = cluster_means / num_clusters.unsqueeze(dim=-1)
        self.mu.data.copy_(cluster_means)
        # self.mu.data = self.mu.cpu()

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
            print("\nEpoch %d at %s" % (epoch, str(datetime.datetime.now())))
            for batch_idx, input_batch in enumerate(tqdm(train_loader, desc="Semi supervised learning", total=len(train_loader))):
                # semi-supervised phase
                image_batch = Variable(input_batch[1]).to(self.device)
                text_batch = Variable(input_batch[2]).to(self.device)
                text_len_batch = Variable(input_batch[3]).to(self.device)
                target_batch = Variable(input_batch[4]).to(self.device)
                optimizer.zero_grad()
                _z = self.forward(image_batch, text_batch, text_len_batch)
                qbatch = self.soft_assignemt(_z)
                semi_loss = self.semi_loss_function(target_batch, qbatch)
                semi_train_loss += semi_loss.data * len(target_batch)
                semi_loss.backward()
                optimizer.step()
                del image_batch, text_batch, text_len_batch, target_batch, _z

            # update p considering short memory
            q = []
            print("\nUpdating p-value at %s" % (str(datetime.datetime.now())))
            for batch_idx, input_batch in enumerate(tqdm(full_loader, total=len(full_loader))):
                # clustering phase
                image_batch = Variable(input_batch[1]).to(self.device)
                text_batch = Variable(input_batch[2]).to(self.device)
                text_len_batch = Variable(input_batch[3]).to(self.device)

                _z = self.forward(image_batch, text_batch, text_len_batch)
                _q = 1.0 / (1.0 + torch.sum((_z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
                _q = _q ** (self.alpha + 1.0) / 2.0
                q.append(_q.data.cpu())
                del image_batch, text_batch, text_len_batch, _z, _q

            q = torch.cat(q, dim=0)
            q = q / torch.sum(q, dim=1, keepdim=True)
            p = self.target_distribution(q)

            adjust_learning_rate(lr * kappa, optimizer)

            print("\nUnsupervised learning at %s" % (str(datetime.datetime.now())))
            for batch_idx, input_batch in enumerate(tqdm(full_loader, total=len(full_loader))):
                # clustering phase
                image_batch = Variable(input_batch[1]).to(self.device)
                text_batch = Variable(input_batch[2]).to(self.device)
                text_len_batch = Variable(input_batch[3]).to(self.device)
                pbatch = p[batch_idx * batch_size: min((batch_idx + 1) * batch_size, full_num)]

                p_inputs = Variable(pbatch).to(self.device)

                _z = self.forward(image_batch, text_batch, text_len_batch)
                qbatch = self.soft_assignemt(_z)
                loss = self.loss_function(p_inputs, qbatch)
                train_loss += loss.data * len(p_inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del image_batch, text_batch, text_len_batch, _z

            train_loss = train_loss / full_num
            semi_train_loss = semi_train_loss / train_num

            train_pred = torch.argmax(p, dim=1).detach().numpy()
            df_pred = pd.DataFrame(data=train_pred, index=short_codes, columns=['pred'])
            df_pred = df_pred.loc[df_train.index]
            train_pred = df_pred['pred']
            train_acc = accuracy_score(train_labels, train_pred)
            train_nmi = normalized_mutual_info_score(train_labels, train_pred, average_method='geometric')
            train_f_1 = f1_score(train_labels, train_pred, average='macro')
            print("\n#Epoch %3d: acc: %.4f, nmi: %.4f, f_1: %.4f, loss: %.4f, semi_loss: %.4f at %s" % (
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

        print("\n\nTesting at %s" % (str(datetime.datetime.now())))
        # update p considering short memory
        test_q = []
        print("\nCalcaulating q-values at %s" % (str(datetime.datetime.now())))
        for batch_idx, input_batch in enumerate(tqdm(full_loader, total=len(full_loader))):
            # clustering phase
            image_batch = Variable(input_batch[1]).to(self.device)
            text_batch = Variable(input_batch[2]).to(self.device)
            text_len_batch = Variable(input_batch[3]).to(self.device)

            _z = self.forward(image_batch, text_batch, text_len_batch)
            _q = 1.0 / (1.0 + torch.sum((_z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
            _q = _q ** (self.alpha + 1.0) / 2.0
            test_q.append(_q.data.cpu())
            del image_batch, text_batch, text_len_batch, _z, _q

        test_short_codes = []
        test_labels = []
        print("\nUpdating p-value at %s" % (str(datetime.datetime.now())))
        for batch_idx, input_batch in enumerate(tqdm(test_loader, total=len(test_loader))):
            test_short_codes.extend(list(input_batch[0]))
            image_batch = Variable(input_batch[1]).to(self.device)
            text_batch = Variable(input_batch[2]).to(self.device)
            text_len_batch = Variable(input_batch[3]).to(self.device)
            test_labels.extend(input_batch[4].tolist())
            _z = self.forward(image_batch, text_batch, text_len_batch)
            _q = 1.0 / (1.0 + torch.sum((_z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
            _q = _q ** (self.alpha + 1.0) / 2.0
            test_q.append(_q.data.cpu())
            del image_batch, text_batch, text_len_batch, _z

        test_q = torch.cat(test_q, dim=0)
        test_q = test_q / torch.sum(test_q, dim=1, keepdim=True)

        test_p = self.target_distribution(test_q)
        test_pred = torch.argmax(test_p, dim=1).detach().numpy()[full_num:]
        test_acc = accuracy_score(test_labels, test_pred)
        test_nmi = normalized_mutual_info_score(test_labels, test_pred, average_method='geometric')
        test_f_1 = f1_score(test_labels, test_pred, average='macro')
        print("\n#Test acc: %.4f, Test nmi: %.4f, Test f_1: %.4f" % (
            test_acc, test_nmi, test_f_1))
        self.acc = test_acc
        self.nmi = test_nmi
        self.f_1 = test_f_1
        if save_path:
            self.save_model(save_path)
    
def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    # self.short_codes[idx], image_tensor, text_tensor, !text_length!, self.label_data[idx]
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1) # [n, 1, c, s]
    attention = attention.view(n, glimpses, -1)
    attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
    weighted = attention * input # [n, g, v, s]
    weighted_mean = weighted.sum(dim=-1) # [n, g, v]
    return weighted_mean.view(n, -1)

def count_percentage(cluster_labels):
    count = dict(collections.Counter(cluster_labels))
    sorted_count = sorted(count.items(), key=lambda x: x[0], reverse=False)
    for cluster in sorted_count:
        print("cluster {} : {:.2%}".format(str(cluster[0]), cluster[1] / len(cluster_labels)))

def adjust_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
