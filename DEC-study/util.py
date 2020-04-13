import collections
import operator

import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
import math
import os
import sys
import numpy as np
import pandas as pd
import _pickle as cPickle
from tqdm import tqdm
import config

torch.manual_seed(42)

def load_pretrain_data(image_dir, token_to_index, df_text_data, df_train, df_test, CONFIG):
    train_index = set(df_train.index)
    train_short_codes = []
    train_text_data = []
    train_label_data = []
    test_index = set(df_test.index)
    test_short_codes = []
    test_text_data = []
    test_label_data = []

    pbar = tqdm(total=df_text_data.shape[0])
    for index, row in df_text_data.iterrows():
        if index in train_index:
            train_short_codes.append(index)
            train_label_data.append(df_train.loc[index][0])
            word_list = df_text_data.loc[index]['caption'].split()
            train_text_data.append(encode_text(word_list, CONFIG.MAX_SENTENCE_LEN, token_to_index))
        elif index in test_index:
            test_short_codes.append(index)
            test_label_data.append(df_test.loc[index][0])
            word_list = df_text_data.loc[index]['caption'].split()
            test_text_data.append(encode_text(word_list, CONFIG.MAX_SENTENCE_LEN, token_to_index))
        pbar.update(1)
    pbar.close()
    train_dataset = LabeledDataset(image_dir, token_to_index, train_short_codes, train_text_data, train_label_data, CONFIG)
    test_dataset = LabeledDataset(image_dir, token_to_index, test_short_codes, test_text_data, test_label_data, CONFIG)
    return train_dataset, test_dataset

def encode_text(word_list, sentence_len, token_to_index):
    vec = torch.zeros(sentence_len).long()
    for i, token in enumerate(word_list):
        if i == sentence_len:
            break
        index = token_to_index.get(token, 0)
        vec[i] = index
    return vec, min(len(word_list), sentence_len)

class LabeledDataset(Dataset):
    def __init__(self, image_dir, token_to_index, short_codes, text_data, label_data, CONFIG):
        self.short_codes = short_codes
        self.text_data = text_data
        self.CONFIG = CONFIG
        self.image_dir = image_dir
        self.label_data = label_data
        self.token_to_index = token_to_index

    def __len__(self):
        return len(self.short_codes)

    def __getitem__(self, idx):
        with open(os.path.join(self.image_dir, self.short_codes[idx]) + '.p', "rb") as f:
            image_data = cPickle.load(f)
        image_tensor = torch.from_numpy(image_data).type(torch.FloatTensor)
        text_tensor = self.text_data[idx][0]
        text_length = self.text_data[idx][1]
        return self.short_codes[idx], image_tensor, text_tensor, text_length, self.label_data[idx]

def load_data(image_dir, token_to_index, df_text_data, df_train, df_test, sample, CONFIG):
    full_short_codes = []
    full_text_data = []
    train_index = set(df_train.index)
    train_short_codes = []
    train_text_data = []
    train_label_data = []
    test_index = set(df_test.index)
    test_short_codes = []
    test_text_data = []
    test_label_data = []
    if sample is not None:
        labeled_index = train_index.union(test_index)
        df_sampled_data = df_text_data.loc[df_text_data.index.difference(labeled_index)].sample(n=sample)
        df_labeled_data = df_text_data.loc[labeled_index]
        df_text_data = pd.concat([df_sampled_data, df_labeled_data])
        df_text_data = df_text_data.sample(frac=1)
    pbar = tqdm(total=df_text_data.shape[0])
    for index, row in df_text_data.iterrows():
        word_list = df_text_data.loc[index]['caption'].split()
        temp = encode_text(word_list, CONFIG.MAX_SENTENCE_LEN, token_to_index)
        if index in train_index:
            full_short_codes.append(index)
            full_text_data.append(temp)
            train_short_codes.append(index)
            train_label_data.append(df_train.loc[index][0])
            train_text_data.append(temp)
        elif index in test_index:
            test_short_codes.append(index)
            test_label_data.append(df_test.loc[index][0])
            test_text_data.append(temp)
        else:
            full_short_codes.append(index)
            full_text_data.append(temp)

        pbar.update(1)
    pbar.close()
    full_dataset = UnlabeledDataset(image_dir, token_to_index, full_short_codes, full_text_data, CONFIG)
    train_dataset = LabeledDataset(image_dir, token_to_index, train_short_codes, train_text_data, train_label_data, CONFIG)
    test_dataset = LabeledDataset(image_dir, token_to_index, test_short_codes, test_text_data, test_label_data, CONFIG)
    return full_dataset, train_dataset, test_dataset, full_short_codes, (train_short_codes, train_label_data), (test_short_codes, test_label_data)

class UnlabeledDataset(Dataset):
    def __init__(self, image_dir, token_to_index, short_codes, text_data, CONFIG):
        self.short_codes = short_codes
        self.text_data = text_data
        self.CONFIG = CONFIG
        self.image_dir = image_dir
        self.token_to_index = token_to_index

    def __len__(self):
        return len(self.short_codes)

    def __getitem__(self, idx):
        with open(os.path.join(self.image_dir, self.short_codes[idx]) + '.p', "rb") as f:
            image_data = cPickle.load(f)
        image_tensor = torch.from_numpy(image_data).type(torch.FloatTensor)
        text_tensor = self.text_data[idx][0]
        text_length = self.text_data[idx][1]
        return self.short_codes[idx], image_tensor, text_tensor, text_length

# def load_pretrain_data(image_dir, token_to_index, df_text_data, df_train, df_test, CONFIG):
#     train_index = set(df_train.index)
#     train_short_codes = []
#     train_text_data = []
#     train_label_data = []
#     test_index = set(df_test.index)
#     test_short_codes = []
#     test_text_data = []
#     test_label_data = []
#
#     pbar = tqdm(total=df_text_data.shape[0])
#     for index, row in df_text_data.iterrows():
#         if index in train_index:
#             train_short_codes.append(index)
#             train_text_data.append(np.array(df_text_data.loc[index]['caption']))
#             train_label_data.append(df_train.loc[index][0])
#         elif index in test_index:
#             test_short_codes.append(index)
#             test_text_data.append(np.array(df_text_data.loc[index]['caption']))
#             test_label_data.append(df_test.loc[index][0])
#         pbar.update(1)
#     pbar.close()
#     train_dataset = PretrainDataset(image_dir, token_to_index, train_short_codes, np.array(train_text_data), train_label_data, CONFIG)
#     test_dataset = PretrainDataset(image_dir, token_to_index, test_short_codes, np.array(test_text_data), test_label_data, CONFIG)
#     return train_dataset, test_dataset
#
#
# class PretrainDataset(Dataset):
#     def __init__(self, image_dir, token_to_index, short_codes, text_data, label_data, CONFIG):
#         self.short_codes = short_codes
#         self.text_data = text_data
#         self.CONFIG = CONFIG
#         self.image_dir = image_dir
#         self.label_data = label_data
#         self.token_to_index = token_to_index
#
#     def __len__(self):
#         return len(self.short_codes)
#
#     def __getitem__(self, idx):
#         with open(os.path.join(self.image_dir, self.short_codes[idx]), "rb") as f:
#             image_data = cPickle.load(f)
#         image_tensor = torch.from_numpy(image_data).type(torch.FloatTensor)
#         text_tensor = self._encode_text(self.text_data[idx])
#         return self.short_codes[idx], image_tensor, text_tensor, self.label_data[idx]
#
#     def _encode_text(self, text):
#         """ Turn a question into a vector of indices and a question length """
#         vec = torch.zeros(self.CONFIG.MAX_SENTENCE_LEN).type(torch.LongTensor)
#         word_list = text.split()
#         for i, token in enumerate(word_list):
#             index = self.token_to_index.get(token, 0)
#             vec[i] = index
#         return vec, len(word_list)