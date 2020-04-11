import collections
import datetime
import os

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

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
            nn.Linear(2048, z_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, n_classes),
            nn.Sigmoid(),
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

    def classify(self, z):
        log_prob = self.classifier(z)
        return log_prob

    def fit(self, train_dataset, test_dataset, lr=0.001, batch_size=128, num_epochs=10, save_path=None):
        trainloader = DataLoader(train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        criterion = nn.NLLLoss().to(self.device)
        self.to(self.device)
        self.train()
        for epoch in range(num_epochs):
            # train 1 epoch
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
            print("#Epoch %3d: acc: %.4f, nmi: %.4f, f_1: %.4f, loss: %.4f at %s" % (
                epoch + 1, train_acc, train_nmi, train_f_1, train_loss, str(datetime.datetime.now())))
        if save_path:
            self.save_model(save_path)

class DDEC(nn.Module):
    def __init__(self, device, pretrained_model, n_classes=10, z_dim=1024, alpha=1.):
        super(self.__class__, self).__init__()
        self.device = device
        self.dualnet = pretrained_model
        self.weight_calculator1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=7, stride=3),
            nn.BatchNorm1d(10),
            nn.ReLU()
        )
        self.weight_calculator2 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=5, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU()
        )
        self.weight_calculator3 = nn.Sequential(
            nn.Linear(in_features=20, out_features=2),
            nn.Sigmoid()
        )
        self.mu = Parameter(torch.Tensor(n_classes, z_dim))
        self.softmax = nn.Softmax(dim=1)
        self.n_classes = n_classes
        self.alpha = alpha
        self.tau = 1

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, image_x, text_x):
        z = self.dualnet(image_x, text_x)
        return z

    def soft_assignemt(self, image_z, text_z):
        q = 1.0 / (1.0 + torch.sum((image_z.unsqueeze(1) - self.image_encoder.mu) ** 2, dim=2) / self.alpha)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)

        r = 1.0 / (1.0 + torch.sum((text_z.unsqueeze(1) - self.text_encoder.mu) ** 2, dim=2) / self.alpha)
        r = r ** (self.alpha + 1.0) / 2.0
        r = r / torch.sum(r, dim=1, keepdim=True)
        return q, r

    def loss_function(self, p, q, r):
        h = torch.mean(p, dim=0, keepdim=True)
        u = torch.full_like(h, fill_value=1 / h.size()[1])
        image_loss = F.kl_div(q.log(), p, reduction='batchmean') + F.kl_div(u.log(), h, reduction='batchmean')
        text_loss = F.kl_div(r.log(), p, reduction='batchmean') + F.kl_div(u.log(), h, reduction='batchmean')
        loss = image_loss + text_loss
        return loss

    def target_distribution(self, q, r, w):
        p_image = q ** 2 / torch.sum(q, dim=0)
        p_image = p_image / torch.sum(p_image, dim=1, keepdim=True)
        p_text = r ** 2 / torch.sum(r, dim=0)
        p_text = p_text / torch.sum(p_text, dim=1, keepdim=True)
        p = torch.stack([p_image, p_text], dim=1)
        p = torch.bmm(w.unsqueeze(dim=1), p).squeeze(dim=1)
        return p

    def fit(self, X, lr=0.001, batch_size=256, num_epochs=10, save_path=None):
        num = len(X)
        num_batch = int(math.ceil(1.0 * len(X) / batch_size))
        '''X: tensor data'''
        self.to(self.device)
        print("=====Training DEC=======")
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)

        print("Extracting initial features at %s" % (str(datetime.datetime.now())))
        z = []
        for batch_idx in range(num_batch):
            image_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][1]
            text_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][2]
            image_inputs = Variable(image_batch).to(self.device)
            text_inputs = Variable(text_batch).to(self.device)
            _z = self.forward(image_inputs, text_inputs)
            z.append(_z.data.cpu())
            del image_batch, text_batch, image_inputs, text_inputs, _z
            torch.cuda.empty_cache()
        z = torch.cat(z, dim=0)
        self.mu.data.copy_(torch.Tensor(image_means))
        self.mu.data = self.mu.cpu()
        self.train()
        best_loss = 99999.
        best_epoch = 0

        for epoch in range(num_epochs):
            # update the target distribution p

            image_z = []
            text_z = []
            w = []
            for batch_idx in range(num_batch):
                image_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][1]
                text_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][2]
                image_inputs = Variable(image_batch).to(self.device)
                text_inputs = Variable(text_batch).to(self.device)
                _image_z, _text_z, _w = self.forward(image_inputs, text_inputs)
                image_z.append(_image_z.data.cpu())
                text_z.append(_text_z.data.cpu())
                w.append(_w.data.cpu())
                del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z, _w
                torch.cuda.empty_cache()
            image_z = torch.cat(image_z, dim=0)
            text_z = torch.cat(text_z, dim=0)
            w = torch.cat(w, dim=0)

            q, r = self.soft_assignemt(image_z, text_z)
            p = self.target_distribution(q, r, w).data
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

                image_z, text_z, _ = self.forward(image_inputs, text_inputs)
                qbatch, rbatch = self.soft_assignemt(image_z.cpu(), text_z.cpu())
                loss = self.loss_function(target, qbatch, rbatch)
                train_loss += loss.data * len(target)
                loss.backward()
                optimizer.step()

                del image_batch, text_batch, image_inputs, text_inputs, image_z, text_z, _
                torch.cuda.empty_cache()
            train_loss = train_loss / num
            if best_loss > train_loss:
                best_loss = train_loss
                best_epoch = epoch
                if save_path:
                    self.save_model(os.path.join(save_path, "socialdec_" + str(self.image_encoder.z_dim)) + '_' + str(
                        self.n_clusters) + ".pt")
            print("#Epoch %3d: Loss: %.4f Best Loss: %.4f at %s" % (
                epoch + 1, train_loss, best_loss, str(datetime.datetime.now())))

        print("#Best Epoch %3d: Best Loss: %.4f" % (
            best_epoch, best_loss))

    def fit_predict(self, X, batch_size=256):
        num = len(X)
        num_batch = int(math.ceil(1.0 * len(X) / batch_size))
        self.to(self.device)
        self.image_encoder.mu.data = self.image_encoder.mu.cpu()
        self.text_encoder.mu.data = self.text_encoder.mu.cpu()

        self.eval()
        image_z = []
        text_z = []
        w = []
        short_codes = []
        for batch_idx in range(num_batch):
            short_codes.append(X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][0])
            image_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][1]
            text_batch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)][2]
            image_inputs = Variable(image_batch).to(self.device)
            text_inputs = Variable(text_batch).to(self.device)
            _image_z, _text_z, _w = self.forward(image_inputs, text_inputs)
            image_z.append(_image_z.data.cpu())
            text_z.append(_text_z.data.cpu())
            w.append(_w.data.cpu())
            del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z, _w
            torch.cuda.empty_cache()
        short_codes = np.concatenate(short_codes, axis=0)
        image_z = torch.cat(image_z, dim=0)
        text_z = torch.cat(text_z, dim=0)
        w = torch.cat(w, dim=0)

        q, r = self.soft_assignemt(image_z, text_z)
        p = self.target_distribution(q, r, w).data
        # y_pred = torch.argmax(p, dim=1).numpy()
        y_confidence, y_pred = torch.max(p, dim=1)
        y_confidence = y_confidence.numpy()
        y_pred = y_pred.numpy()
        p = p.numpy()
        w = w.data.numpy()
        count_percentage(y_pred)
        return short_codes, y_pred, y_confidence, p, w
    
def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    # self.short_codes[idx], image_tensor, text_tensor, !text_length!, self.label_data[idx]
    batch.sort(key=lambda x: x[-2], reverse=True)
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