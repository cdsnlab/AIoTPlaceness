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


class WeightCalc(nn.Module):
    def __init__(self, device, ours=False, use_prior=False, input_dim=300, n_clusters=10, alpha=1):
        super(self.__class__, self).__init__()
        self.device = device
        self.ours = ours
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.use_prior = use_prior
        if use_prior:
            self.prior = torch.zeros(self.n_clusters).float().to(device)

        self.input_dim = input_dim
        self.n_clusters = n_clusters

        self.layer0 = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        self.acc = 0.
        self.nmi = 0.
        self.f_1 = 0.
        self.softmax = nn.Softmax(dim=1)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, image_input, text_input):
        output0 = self.layer0(torch.cat([image_input, text_input], dim=1))
        weight = self.layer1(output0)
        return weight

    def probabililty_fusion(self, q, r, image_input, text_input):
        #s = self.weight_parameter.expand_as(q) * q + (1 - self.weight_parameter).expand_as(r) * r
        w = self.forward(image_input, text_input)
        s = w.expand_as(q) * q + (1-w.expand_as(r)) * r
        return s, w

    def semi_loss_function(self, label_batch, s_batch):
        supervised_loss = F.nll_loss(s_batch.log(), label_batch)
        return supervised_loss

    def target_distribution(self, q):
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit_predict(self, mdec, full_dataset, train_dataset, test_dataset, args, CONFIG, lr=0.001, batch_size=256, num_epochs=10, update_time=1, save_path=None, tol=1e-3, kappa=0.1):
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
        full_short_codes = full_dataset[:][0]
        train_short_codes = train_dataset[:][0]
        test_short_codes = test_dataset[:][0]
        train_labels = train_dataset[:][3].squeeze(dim=0).data.cpu().numpy()
        test_labels = test_dataset[:][3].squeeze(dim=0).data.cpu().numpy()
        df_train = pd.DataFrame(data=train_labels, index=train_short_codes, columns=['label'])
        df_test = pd.DataFrame(data=test_labels, index=test_short_codes, columns=['label'])

        if self.use_prior:
            for label in train_labels:
                self.prior[label] = self.prior[label] + 1
            self.prior /= len(train_labels)

        print("Calculating initial p at %s" % (str(datetime.datetime.now())))
        # update p considering short memory
        s = []
        for batch_idx in range(test_num_batch):
            image_batch = test_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, test_num)][1]
            text_batch = test_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, test_num)][2]

            image_inputs = Variable(image_batch).to(self.device)
            text_inputs = Variable(text_batch).to(self.device)

            _image_z, _text_z = mdec.forward(image_inputs, text_inputs)
            _q, _r = mdec.soft_assignemt(_image_z, _text_z)
            _q = Variable(torch.Tensor(_q.data.cpu().numpy())).to(self.device)
            _r = Variable(torch.Tensor(_r.data.cpu().numpy())).to(self.device)
            _s, _ = self.probabililty_fusion(_q, _r, image_inputs, text_inputs)
            s.append(_s.data.cpu())

            del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z, _q, _r, _s
        s = torch.cat(s, dim=0)

        initial_pred = torch.argmax(s, dim=1).numpy()
        initial_acc = accuracy_score(test_labels, initial_pred)
        initial_nmi = normalized_mutual_info_score(test_labels, initial_pred, average_method='geometric')
        initial_f_1 = f1_score(test_labels, initial_pred, average='macro')
        print("#Initial measure: acc: %.4f, nmi: %.4f, f_1: %.4f" % (initial_acc, initial_nmi, initial_f_1))

        flag_end_training = False
        for epoch in range(num_epochs):
            print("Epoch %d at %s" % (epoch, str(datetime.datetime.now())))
            # update the target distribution p
            self.train()
            # train 1 epoch
            train_supervised_loss = 0.0
            # update p considering short memory
            s = []
            for batch_idx in range(train_num_batch):
                image_batch = train_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, train_num)][1]
                text_batch = train_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, train_num)][2]
                label_batch = train_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, train_num)][3].squeeze(dim=0)

                image_inputs = Variable(image_batch).to(self.device)
                text_inputs = Variable(text_batch).to(self.device)
                label_inputs = Variable(label_batch).to(self.device)

                _image_z, _text_z = mdec.forward(image_inputs, text_inputs)
                _q, _r = mdec.soft_assignemt(_image_z, _text_z)
                _q = Variable(torch.Tensor(_q.data.cpu().numpy())).to(self.device)
                _r = Variable(torch.Tensor(_r.data.cpu().numpy())).to(self.device)
                _s, _ = self.probabililty_fusion(_q, _r, image_inputs, text_inputs)
                supervised_loss = self.semi_loss_function(label_inputs, _s)
                optimizer.zero_grad()
                supervised_loss.backward()
                optimizer.step()
                train_supervised_loss += supervised_loss.data * len(label_inputs)
                s.append(_s.data.cpu())

                del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z, _q, _r, _s
            s = torch.cat(s, dim=0)
            train_supervised_loss /= train_num

            train_pred = torch.argmax(s, dim=1).numpy()
            train_acc = accuracy_score(train_labels, train_pred)
            train_nmi = normalized_mutual_info_score(train_labels, train_pred, average_method='geometric')
            train_f_1 = f1_score(train_labels, train_pred, average='macro')
            print("#Train measure %3d: acc: %.4f, nmi: %.4f, f_1: %.4f" % (
                epoch + 1, train_acc, train_nmi, train_f_1))
            print("#Train loss %3d: super lss: %.4f" % (
                epoch + 1, train_supervised_loss))
            if epoch == 0:
                train_pred_last = train_pred
            else:
                delta_label = np.sum(train_pred != train_pred_last).astype(np.float32) / len(train_pred)
                train_pred_last = train_pred
                if delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    flag_end_training = True

            self.eval()
            test_supervised_loss = 0.0
            # update p considering short memory
            s = []
            w = []
            q = []
            r = []
            for batch_idx in range(test_num_batch):
                image_batch = test_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, test_num)][1]
                text_batch = test_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, test_num)][2]
                label_batch = test_dataset[batch_idx * batch_size: min((batch_idx + 1) * batch_size, test_num)][3].squeeze(dim=0)

                image_inputs = Variable(image_batch).to(self.device)
                text_inputs = Variable(text_batch).to(self.device)
                label_inputs = Variable(label_batch).to(self.device)

                _image_z, _text_z = mdec.forward(image_inputs, text_inputs)
                _q, _r = mdec.soft_assignemt(_image_z, _text_z)
                _q = Variable(torch.Tensor(_q.data.cpu().numpy())).to(self.device)
                _r = Variable(torch.Tensor(_r.data.cpu().numpy())).to(self.device)
                _s, _w = self.probabililty_fusion(_q, _r, image_inputs, text_inputs)
                supervised_loss = self.semi_loss_function(label_inputs, _s)
                test_supervised_loss += supervised_loss.data * len(label_inputs)
                s.append(_s.data.cpu())
                w.append(_w.data.cpu())
                q.append(_q.data.cpu())
                r.append(_r.data.cpu())


                del image_batch, text_batch, image_inputs, text_inputs, _image_z, _text_z, _q, _r, _s
            s = torch.cat(s, dim=0)
            w = torch.cat(w, dim=0)
            q = torch.cat(q, dim=0)
            r = torch.cat(r, dim=0)
            test_supervised_loss /= test_num

            x = torch.mean(torch.stack([q, r]), dim=0)
            df_test = pd.DataFrame(data=np.column_stack([torch.argmax(x, dim=1).numpy(), test_labels]),
                                   index=test_short_codes, columns=['pred', 'label'])
            df_test.to_csv('xdec_label.csv', encoding='utf-8-sig')
            df_test_x = pd.DataFrame(data=x.data.numpy(), index=test_short_codes)
            df_test_x.to_csv('xdec_x.csv', encoding='utf-8-sig')

            test_pred = torch.argmax(s, dim=1).numpy()
            test_acc = accuracy_score(test_labels, test_pred)
            test_nmi = normalized_mutual_info_score(test_labels, test_pred, average_method='geometric')
            test_f_1 = f1_score(test_labels, test_pred, average='macro')
            df_test = pd.DataFrame(data=np.column_stack([torch.argmax(s, dim=1).numpy(), test_labels]), index=test_short_codes, columns=['pred', 'label'])
            df_test.to_csv('onedec_label.csv', encoding='utf-8-sig')
            df_test_s = pd.DataFrame(data=s.data.numpy(), index=test_short_codes)
            df_test_s.to_csv('onedec_s.csv', encoding='utf-8-sig')
            df_test_w = pd.DataFrame(data=w.data.numpy(), index=test_short_codes)
            df_test_w.to_csv('onedec_w.csv', encoding='utf-8-sig')
            df_test_q = pd.DataFrame(data=q.data.numpy(), index=test_short_codes)
            df_test_q.to_csv('onedec_q.csv', encoding='utf-8-sig')
            df_test_r = pd.DataFrame(data=r.data.numpy(), index=test_short_codes)
            df_test_r.to_csv('onedec_r.csv', encoding='utf-8-sig')
            print("#Test measure %3d: acc: %.4f, nmi: %.4f, f_1: %.4f" % (
                epoch + 1, test_acc, test_nmi, test_f_1))
            print("#Test loss %3d: super lss: %.4f" % (
                epoch + 1, test_supervised_loss))
            self.acc = test_acc
            self.nmi = test_nmi
            self.f_1 = test_f_1

            if flag_end_training:
                break

        if save_path and not args.resume:
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
