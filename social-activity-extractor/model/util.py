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

torch.manual_seed(42)

def load_multi_csv_data(df_image_data, df_text_data, df_weight, df_train, df_val, CONFIG):
    train_index = set(df_train.index)
    val_index = set(df_val.index)
    train_short_codes = []
    train_image_data = []
    train_text_data = []
    train_label_data = []
    train_weight_data = []
    val_short_codes = []
    val_image_data = []
    val_text_data = []
    val_label_data = []
    val_weight_data = []

    pbar = tqdm(total=df_image_data.shape[0])
    for index, row in df_image_data.iterrows():
        if index in train_index:
            train_short_codes.append(index)
            train_image_data.append(np.array(row))
            train_text_data.append(np.array(df_text_data.loc[index]))
            train_label_data.append(df_train.loc[index][0])
            train_weight_data.append(np.array(df_weight.loc[index], dtype=np.float32))
        elif index in val_index:
            val_short_codes.append(index)
            val_image_data.append(np.array(row))
            val_text_data.append(np.array(df_text_data.loc[index]))
            val_label_data.append(df_val.loc[index][0])
            val_weight_data.append(np.array(df_weight.loc[index], dtype=np.float32))
        pbar.update(1)
    pbar.close()
    train_dataset = LabeledWeightedMultiCSVDataset(train_short_codes, np.array(train_image_data), np.array(train_text_data), train_label_data, np.array(train_weight_data, dtype=np.float32), CONFIG)
    val_dataset = LabeledWeightedMultiCSVDataset(val_short_codes, np.array(val_image_data), np.array(val_text_data), val_label_data, np.array(val_weight_data, dtype=np.float32), CONFIG)
    return train_dataset, val_dataset

class LabeledWeightedMultiCSVDataset(Dataset):
    def __init__(self, short_codes, image_data, text_data, label_data, weight_data, CONFIG):
        self.short_codes = short_codes
        self.image_data = image_data
        self.text_data = text_data
        self.CONFIG = CONFIG
        self.label_data = label_data
        self.weight_data = weight_data

    def __len__(self):
        return len(self.short_codes)

    def __getitem__(self, idx):
        image_tensor = torch.from_numpy(self.image_data[idx]).type(torch.FloatTensor)
        text_tensor = torch.from_numpy(self.text_data[idx]).type(torch.FloatTensor)
        return self.short_codes[idx], image_tensor, text_tensor, self.label_data[idx], self.weight_data[idx]

def load_semi_supervised_csv_data(df_image_data, df_text_data, df_train, df_val, df_weight, CONFIG):
    train_index = set(df_train.index)
    val_index = set(df_val.index)
    full_short_codes = []
    full_image_data = []
    full_text_data = []
    train_short_codes = []
    train_image_data = []
    train_text_data = []
    train_label_data = []
    train_weight_data = []
    val_short_codes = []
    val_image_data = []
    val_text_data = []
    val_label_data = []
    val_weight_data = []

    pbar = tqdm(total=df_image_data.shape[0])
    for index, row in df_image_data.iterrows():
        if index in train_index:
            full_short_codes.append(index)
            full_image_data.append(np.array(row))
            full_text_data.append(np.array(df_text_data.loc[index]))
            train_short_codes.append(index)
            train_image_data.append(np.array(row))
            train_text_data.append(np.array(df_text_data.loc[index]))
            train_label_data.append(df_train.loc[index][0])
            train_weight_data.append(np.array(df_weight.loc[index]))
        elif index in val_index:
            val_short_codes.append(index)
            val_image_data.append(np.array(row))
            val_text_data.append(np.array(df_text_data.loc[index]))
            val_label_data.append(df_val.loc[index][0])
            val_weight_data.append(np.array(df_weight.loc[index]))
        else:
            full_short_codes.append(index)
            full_image_data.append(np.array(row))
            full_text_data.append(np.array(df_text_data.loc[index]))
        pbar.update(1)
    pbar.close()
    full_dataset = MultiCSVDataset(full_short_codes, np.array(full_image_data), np.array(full_text_data), CONFIG)
    train_dataset = LabeledMultiCSVDataset(train_short_codes, np.array(train_image_data), np.array(train_text_data), train_label_data, np.array(train_weight_data), CONFIG)
    val_dataset = LabeledMultiCSVDataset(val_short_codes, np.array(val_image_data), np.array(val_text_data), val_label_data, np.array(val_weight_data), CONFIG)
    return full_dataset, train_dataset, val_dataset

class MultiCSVDataset(Dataset):
    def __init__(self, short_codes, image_data, text_data, CONFIG):
        self.short_codes = short_codes
        self.image_data = image_data
        self.text_data = text_data
        self.CONFIG = CONFIG

    def __len__(self):
        return len(self.short_codes)

    def __getitem__(self, idx):
        image_tensor = torch.from_numpy(self.image_data[idx]).type(torch.FloatTensor)
        text_tensor = torch.from_numpy(self.text_data[idx]).type(torch.FloatTensor)
        return self.short_codes[idx], image_tensor, text_tensor

class LabeledMultiCSVDataset(Dataset):
    def __init__(self, short_codes, image_data, text_data, label_data, weight_data, CONFIG):
        self.short_codes = short_codes
        self.image_data = image_data
        self.text_data = text_data
        self.CONFIG = CONFIG
        self.label_data = label_data
        self.weight_data = weight_data

    def __len__(self):
        return len(self.short_codes)

    def __getitem__(self, idx):
        image_tensor = torch.from_numpy(self.image_data[idx]).type(torch.FloatTensor)
        text_tensor = torch.from_numpy(self.text_data[idx]).type(torch.FloatTensor)
        label_tensor = torch.LongTensor([self.label_data[idx]])
        weight_tensor = torch.from_numpy(self.weight_data[idx]).type(torch.FloatTensor)
        return self.short_codes[idx], image_tensor, text_tensor, label_tensor, weight_tensor

def load_semi_supervised_uni_csv_data(df_input_data, df_train, df_val, CONFIG):
    train_index = set(df_train.index)
    val_index = set(df_val.index)
    full_short_codes = []
    full_input_data = []
    train_short_codes = []
    train_input_data = []
    train_label_data = []
    val_short_codes = []
    val_input_data = []
    val_label_data = []

    pbar = tqdm(total=df_input_data.shape[0])
    for index, row in df_input_data.iterrows():
        if index in train_index:
            full_short_codes.append(index)
            full_input_data.append(np.array(row))
            train_short_codes.append(index)
            train_input_data.append(np.array(row))
            train_label_data.append(df_train.loc[index][0])
        elif index in val_index:
            val_short_codes.append(index)
            val_input_data.append(np.array(row))
            val_label_data.append(df_val.loc[index][0])
        else:
            full_short_codes.append(index)
            full_input_data.append(np.array(row))
        pbar.update(1)
    pbar.close()
    full_dataset = UniCSVDataset(full_short_codes, np.array(full_input_data), CONFIG)
    train_dataset = LabeledUniCSVDataset(train_short_codes, np.array(train_input_data), train_label_data, CONFIG)
    val_dataset = LabeledUniCSVDataset(val_short_codes, np.array(val_input_data), val_label_data, CONFIG)
    return full_dataset, train_dataset, val_dataset

class UniCSVDataset(Dataset):
    def __init__(self, short_codes, input_data, CONFIG):
        self.short_codes = short_codes
        self.input_data = input_data
        self.CONFIG = CONFIG

    def __len__(self):
        return len(self.short_codes)

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.input_data[idx]).type(torch.FloatTensor)
        return self.short_codes[idx], input_tensor

class LabeledUniCSVDataset(Dataset):
    def __init__(self, short_codes, input_data, label_data, CONFIG):
        self.short_codes = short_codes
        self.input_data = input_data
        self.CONFIG = CONFIG
        self.label_data = label_data

    def __len__(self):
        return len(self.short_codes)

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.input_data[idx]).type(torch.FloatTensor)
        label_tensor = torch.LongTensor(self.label_data[idx])
        return self.short_codes[idx], input_tensor, label_tensor

# def load_semi_supervised_csv_data(df_image_data, df_text_data, df_train, df_val, CONFIG):
#     train_index = set(df_train.index)
#     val_index = set(df_val.index)
#     short_codes = []
#     image_data = []
#     text_data = []
#     train_label_data = []
#     val_label_data = []
#
#     pbar = tqdm(total=df_image_data.shape[0])
#     for index, row in df_image_data.iterrows():
#         short_codes.append(index)
#         image_data.append(np.array(row))
#         text_data.append(np.array(df_text_data.loc[index]))
#         if index in train_index:
#             train_label_data.append(df_train.loc[index][0])
#             val_label_data.append(-1)
#         elif index in val_index:
#             train_label_data.append(-1)
#             val_label_data.append(df_val.loc[index][0])
#         else:
#             train_label_data.append(-1)
#             val_label_data.append(-1)
#         pbar.update(1)
#     pbar.close()
#     full_dataset = SemiSupervisedDataset(short_codes, np.array(image_data), np.array(text_data), train_label_data, val_label_data, CONFIG)
#     return full_dataset
#
# class SemiSupervisedDataset(Dataset):
#     def __init__(self, short_codes, image_data, text_data, train_label, val_label, CONFIG):
#         self.short_codes = short_codes
#         self.image_data = image_data
#         self.text_data = text_data
#         self.CONFIG = CONFIG
#         self.train_label = train_label
#         self.val_label = val_label
#
#     def __len__(self):
#         return len(self.short_codes)
#
#     def __getitem__(self, idx):
#         image_tensor = torch.from_numpy(self.image_data[idx]).type(torch.FloatTensor)
#         text_tensor = torch.from_numpy(self.text_data[idx]).type(torch.FloatTensor)
#         return self.short_codes[idx], image_tensor, text_tensor, self.train_label[idx], self.val_label[idx]

# def load_multi_csv_data(df_image_data, df_text_data, CONFIG):
#     short_codes = []
#     image_data = []
#     text_data = []
#
#     pbar = tqdm(total=df_image_data.shape[0])
#     for index, row in df_image_data.iterrows():
#         short_codes.append(index)
#         image_data.append(np.array(row))
#         text_data.append(np.array(df_text_data.loc[index]))
#         pbar.update(1)
#     pbar.close()
#     full_dataset = MultiCSVDataset(short_codes, np.array(image_data), np.array(text_data), CONFIG)
#     return full_dataset
#
#
# class MultiCSVDataset(Dataset):
#     def __init__(self, short_codes, image_data, text_data, CONFIG):
#         self.short_codes = short_codes
#         self.image_data = image_data
#         self.text_data = text_data
#         self.CONFIG = CONFIG
#
#     def __len__(self):
#         return len(self.short_codes)
#
#     def __getitem__(self, idx):
#         image_tensor = torch.from_numpy(self.image_data[idx]).type(torch.FloatTensor)
#         text_tensor = torch.from_numpy(self.text_data[idx]).type(torch.FloatTensor)
#         return self.short_codes[idx], image_tensor, text_tensor
#
#
# def load_multi_csv_data_with_label(df_image_data, df_text_data, df_label, CONFIG):
#     short_codes = []
#     image_data = []
#     text_data = []
#     label_data = []
#
#     pbar = tqdm(total=df_image_data.shape[0])
#     for index, row in df_image_data.iterrows():
#         short_codes.append(index)
#         image_data.append(np.array(row))
#         text_data.append(np.array(df_text_data.loc[index]))
#         label_data.append(df_label.loc[index][0])
#         pbar.update(1)
#     pbar.close()
#     full_dataset = MultiCSVDataset_with_label(short_codes, np.array(image_data), np.array(text_data), label_data,
#                                               CONFIG)
#     return full_dataset
#
#
# class MultiCSVDataset_with_label(Dataset):
#     def __init__(self, short_codes, image_data, text_data, label_data, CONFIG):
#         self.short_codes = short_codes
#         self.image_data = image_data
#         self.text_data = text_data
#         self.label_data = label_data
#         self.CONFIG = CONFIG
#
#     def __len__(self):
#         return len(self.short_codes)
#
#     def __getitem__(self, idx):
#         image_tensor = torch.from_numpy(self.image_data[idx]).type(torch.FloatTensor)
#         text_tensor = torch.from_numpy(self.text_data[idx]).type(torch.FloatTensor)
#         return self.short_codes[idx], image_tensor, text_tensor, self.label_data[idx]


def load_autoencoder_data(df_input_data, CONFIG):
    df_train, df_val = train_test_split(df_input_data, test_size=0.2, shuffle=True, random_state=42)
    train_short_codes = []
    train_data = []
    pbar = tqdm(total=df_train.shape[0])
    for index, row in df_train.iterrows():
        train_short_codes.append(index)
        train_data.append(np.array(row))
        pbar.update(1)
    pbar.close()
    train_dataset = CSVDataset(train_short_codes, np.array(train_data), CONFIG)

    val_short_codes = []
    val_data = []
    pbar = tqdm(total=df_val.shape[0])
    for index, row in df_val.iterrows():
        val_short_codes.append(index)
        val_data.append(np.array(row))
        pbar.update(1)
    pbar.close()
    val_dataset = CSVDataset(val_short_codes, np.array(val_data), CONFIG)
    return train_dataset, val_dataset

def load_csv_data(args, CONFIG):
    df_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.target_csv), index_col=0,
                          encoding='utf-8-sig')
    df_train, df_val = train_test_split(df_data, test_size=args.split_rate)
    train_short_codes = []
    train_data = []
    pbar = tqdm(total=df_train.shape[0])
    for index, row in df_train.iterrows():
        train_short_codes.append(index)
        train_data.append(np.array(row))
        pbar.update(1)
    pbar.close()
    train_dataset = CSVDataset(train_short_codes, np.array(train_data), CONFIG)

    val_short_codes = []
    val_data = []
    pbar = tqdm(total=df_val.shape[0])
    for index, row in df_val.iterrows():
        val_short_codes.append(index)
        val_data.append(np.array(row))
        pbar.update(1)
    pbar.close()
    val_dataset = CSVDataset(val_short_codes, np.array(val_data), CONFIG)
    return train_dataset, val_dataset

class CSVDataset(Dataset):
    def __init__(self, short_codes, data, CONFIG):
        self.short_codes = short_codes
        self.data = data
        self.CONFIG = CONFIG

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.data[idx]).type(torch.FloatTensor)
        return self.short_codes[idx], input_tensor


def load_text_data(df_full, df_train_label, df_val_label, CONFIG, word2idx, n_clusters, de=None):
    df_train_index = set(df_train_label.index)
    df_val_index = set(df_val_label.index)
    train_short_codes = []
    train_input_data = []
    train_label_data = []
    val_short_codes = []
    val_input_data = []
    val_label_data = []
    pbar = tqdm(total=df_full.shape[0])
    for index, row in df_full.iterrows():
        pbar.update(1)
        if index in df_train_index:
            train_short_codes.append(index)
            train_input_data.append(row.iloc[0])
            train_label_data.append(df_train_label.loc[index][0])
        elif index in df_val_index:
            val_short_codes.append(index)
            val_input_data.append(row.iloc[0])
            val_label_data.append(df_val_label.loc[index][0])
    pbar.close()
    train_dataset, val_dataset = TextDataset(train_short_codes, train_input_data, train_label_data, CONFIG, word2idx, de, n_clusters=n_clusters), \
                                 TextDataset(val_short_codes, val_input_data, val_label_data, CONFIG, word2idx, de, n_clusters=n_clusters)
    return train_dataset, val_dataset


class TextDataset(Dataset):
    def __init__(self, short_codes, input_data, label_data, CONFIG, word2idx, de=None, n_clusters=12):
        self.short_codes = short_codes
        self.input_data = input_data
        self.label_data = label_data
        self.word2idx = word2idx
        self.CONFIG = CONFIG
        self.n_clusters = n_clusters
        self.de = de

    def __len__(self):
        return len(self.short_codes)

    def __getitem__(self, idx):
        word_list = self.input_data[idx].split()
        index_list = []
        if len(word_list) > self.CONFIG.MAX_SENTENCE_LEN:
            # truncate sentence if sentence length is longer than `max_sentence_len`
            word_list = word_list[:self.CONFIG.MAX_SENTENCE_LEN]
            word_list[-1] = '<EOS>'
        else:
            word_list = word_list + ['<PAD>'] * (self.CONFIG.MAX_SENTENCE_LEN - len(word_list))
        for word in word_list:
            index_list.append(self.word2idx[word])
        text_array = np.array(index_list)
        text_tensor = torch.from_numpy(text_array).type(torch.LongTensor)
        label_tensor = torch.from_numpy(np.array(self.label_data[idx])).type(torch.LongTensor)
        if self.de is not None:
            de_data = np.zeros(self.n_clusters)
            for word in word_list:
                if word in self.de:
                    dic_label = self.de[word]
                    de_data[dic_label] = de_data[dic_label] + 1
            de_tensor = torch.from_numpy(de_data).type(torch.FloatTensor)
            return self.short_codes[idx], text_tensor, label_tensor, de_tensor
        else:
            return self.short_codes[idx], text_tensor, label_tensor


# def load_text_data_with_short_code(args, CONFIG, word2idx):
#     full_data = []
#     df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'posts.csv'), header=None,
#                           encoding='utf-8-sig')
#     pbar = tqdm(total=df_data.shape[0])
#     for index, row in df_data.iterrows():
#         pbar.update(1)
#         short_code = row.iloc[0]
#         text_data = row.iloc[1]
#         full_data.append([text_data, short_code])
#         del text_data
#     pbar.close()
#     full_dataset = TextDataset_with_short_code(full_data, CONFIG, word2idx)
#     return full_dataset
#
#
# class TextDataset_with_short_code(Dataset):
#     def __init__(self, data_list, CONFIG, word2idx):
#         self.data = data_list
#         self.word2idx = word2idx
#         self.CONFIG = CONFIG
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         word_list = self.data[idx][0].split()
#         index_list = []
#         if len(word_list) > self.CONFIG.MAX_SENTENCE_LEN:
#             # truncate sentence if sentence length is longer than `max_sentence_len`
#             word_list = word_list[:self.CONFIG.MAX_SENTENCE_LEN]
#             word_list[-1] = '<EOS>'
#         else:
#             word_list = word_list + ['<PAD>'] * (self.CONFIG.MAX_SENTENCE_LEN - len(word_list))
#         for word in word_list:
#             index_list.append(self.word2idx[word])
#         text_array = np.array(index_list)
#         text_tensor = torch.from_numpy(text_array).type(torch.LongTensor)
#         return text_tensor, self.data[idx][1]


def load_image_data_with_short_code(args, CONFIG):
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    full_data = []
    df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'posts.csv'), header=None,
                          encoding='utf-8-sig')
    pbar = tqdm(total=df_data.shape[0])
    for index, row in df_data.iterrows():
        pbar.update(1)
        short_code = row.iloc[0]
        image_path = row.iloc[2].replace('/mnt/SEOUL_SUBWAY_DATA/', '/ssdmnt/placeness/SEOUL_SUBWAY_DATA_300x300/')
        full_data.append([short_code, image_path])
    pbar.close()
    full_dataset = ImageDataset_with_short_code(full_data, CONFIG, img_transform)
    return full_dataset


class ImageDataset_with_short_code(Dataset):
    def __init__(self, data_list, CONFIG, transform):
        self.data = data_list
        self.CONFIG = CONFIG
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_tensor = self.transform(pil_loader(self.data[idx][1]))
        return self.data[idx][0], image_tensor


def load_image_data(df_full, df_train_label, df_val_label, CONFIG):
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    df_train_index = set(df_train_label.index)
    df_val_index = set(df_val_label.index)
    train_short_codes = []
    train_input_data = []
    train_label_data = []
    val_short_codes = []
    val_input_data = []
    val_label_data = []
    pbar = tqdm(total=df_full.shape[0])
    for index, row in df_full.iterrows():
        pbar.update(1)
        if index in df_train_index:
            train_short_codes.append(index)
            image_path = row.iloc[1].replace('/mnt/SEOUL_SUBWAY_DATA/', '/ssdmnt/placeness/SEOUL_SUBWAY_DATA_300x300/')
            #train_input_data.append(image_path)
            image_data = img_transform(pil_loader(image_path))
            train_input_data.append(image_data)
            train_label_data.append(df_train_label.loc[index][0])
        elif index in df_val_index:
            val_short_codes.append(index)
            image_path = row.iloc[1].replace('/mnt/SEOUL_SUBWAY_DATA/', '/ssdmnt/placeness/SEOUL_SUBWAY_DATA_300x300/')
            #val_input_data.append(image_path)
            image_data = img_transform(pil_loader(image_path))
            val_input_data.append(image_data)
            val_label_data.append(df_val_label.loc[index][0])
    pbar.close()
    train_dataset, val_dataset = ImageDataset(train_short_codes, train_input_data, train_label_data, img_transform, CONFIG), \
                                 ImageDataset(val_short_codes, val_input_data, val_label_data, img_transform, CONFIG)
    return train_dataset, val_dataset


class ImageDataset(Dataset):
    def __init__(self, short_codes, input_data, label_data, transform, CONFIG):
        self.short_codes = short_codes
        self.input_data = input_data
        self.label_data = label_data
        self.CONFIG = CONFIG
        self.transform = transform

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        # image_tensor = self.transform(pil_loader(self.input_data[idx]))
        image_tensor = torch.from_numpy(np.array(self.image_data[idx])).type(torch.FloatTensor)
        label_tensor = torch.from_numpy(np.array(self.label_data[idx])).type(torch.LongTensor)
        return self.short_codes[idx], image_tensor, label_tensor


def load_image_pretrain_data(args, CONFIG):
    full_data = []
    dataset_path = os.path.join(CONFIG.DATA_PATH, 'dataset', args.target_dataset)
    image_dir = os.path.join(dataset_path, 'original')
    count = 0
    for image_path in tqdm(os.listdir(image_dir)):
        with open(os.path.join(image_dir, image_path), "rb") as f:
            image_data = cPickle.load(f)
        for image in image_data:
            full_data.append(image)
        f.close()
        del image_data
        if count > 5000:
            break
        count = count + 1
    train_size = int(args.split_rate * len(full_data))
    val_size = len(full_data) - train_size
    train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])
    train_dataset, val_dataset = Image_pretrain_Dataset(train_data, CONFIG), \
                                 Image_pretrain_Dataset(val_data, CONFIG)
    return train_dataset, val_dataset


class Image_pretrain_Dataset(Dataset):
    def __init__(self, data_list, CONFIG):
        self.data = data_list
        self.CONFIG = CONFIG

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_tensor = torch.from_numpy(self.data[idx]).type(torch.FloatTensor)
        return image_tensor


def load_imgseq_data(args, CONFIG):
    full_data = []
    print("Using embedding model: ", args.arch)
    image_dir = os.path.join(CONFIG.DATASET_PATH, args.target_dataset, args.arch)
    for image_path in tqdm(os.listdir(image_dir)):
        with open(os.path.join(image_dir, image_path), "rb") as f:
            image_data = cPickle.load(f)
        full_data.append(image_data)
        f.close()
        del image_data
    full_data = np.array(full_data, dtype=np.float32)
    train_size = int(args.split_rate * len(full_data))
    val_size = len(full_data) - train_size
    train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])
    train_dataset, val_dataset = ImgseqDataset(train_data, CONFIG), \
                                 ImgseqDataset(val_data, CONFIG)
    return train_dataset, val_dataset


class ImgseqDataset(Dataset):
    def __init__(self, data_list, CONFIG):
        self.data = data_list
        self.CONFIG = CONFIG

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imgseq_tensor = torch.from_numpy(self.data[idx]).type(torch.FloatTensor)
        return imgseq_tensor


def load_multimodal_data(args, CONFIG, word2idx):
    full_data = []
    df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'posts.csv'), header=None,
                          encoding='utf-8-sig')
    image_dir = os.path.join(CONFIG.DATASET_PATH, args.target_dataset, args.arch)
    pbar = tqdm(total=df_data.shape[0])
    for index, row in df_data.iterrows():
        pbar.update(1)
        text_data = row.iloc[1]
        image_path = os.path.join(image_dir, row.iloc[0]) + '.p'
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_data = cPickle.load(f)
            full_data.append([text_data, image_data])
            del text_data, image_data
        else:
            del text_data
            continue
    pbar.close()
    train_size = int(args.split_rate * len(full_data))
    val_size = len(full_data) - train_size
    train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])
    train_dataset, val_dataset = MultimodalDataset(train_data, CONFIG, word2idx), \
                                 MultimodalDataset(val_data, CONFIG, word2idx)
    return train_dataset, val_dataset


class MultimodalDataset(Dataset):
    def __init__(self, data_list, CONFIG, word2idx):
        self.data = data_list
        self.word2idx = word2idx
        self.CONFIG = CONFIG

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word_list = self.data[idx][0].split()
        index_list = []
        if len(word_list) > self.CONFIG.MAX_SENTENCE_LEN:
            # truncate sentence if sentence length is longer than `max_sentence_len`
            word_list = word_list[:self.CONFIG.MAX_SENTENCE_LEN]
            word_list[-1] = '<EOS>'
        else:
            word_list = word_list + ['<PAD>'] * (self.CONFIG.MAX_SENTENCE_LEN - len(word_list))
        for word in word_list:
            index_list.append(self.word2idx[word])
        text_array = np.array(index_list)
        text_tensor = torch.from_numpy(text_array).type(torch.LongTensor)
        imgseq_tensor = torch.from_numpy(self.data[idx][1]).type(torch.FloatTensor)

        return text_tensor, imgseq_tensor


def load_fullmultimodal_data(args, CONFIG, word2idx):
    full_data = []
    df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'posts.csv'), header=None,
                          encoding='utf-8-sig')
    image_dir = os.path.join(CONFIG.DATASET_PATH, args.target_dataset, args.arch)
    pbar = tqdm(total=df_data.shape[0])
    for index, row in df_data.iterrows():
        pbar.update(1)
        short_code = row.iloc[0]
        text_data = row.iloc[1]
        image_path = os.path.join(image_dir, row.iloc[0]) + '.p'
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_data = cPickle.load(f)
            full_data.append([text_data, image_data, short_code])
            del text_data, image_data
        else:
            del text_data
            continue
    pbar.close()
    full_dataset = FullMultimodalDataset(full_data, CONFIG, word2idx)
    return full_dataset


class FullMultimodalDataset(Dataset):
    def __init__(self, data_list, CONFIG, word2idx):
        self.data = data_list
        self.word2idx = word2idx
        self.CONFIG = CONFIG

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word_list = self.data[idx][0].split()
        index_list = []
        if len(word_list) > self.CONFIG.MAX_SENTENCE_LEN:
            # truncate sentence if sentence length is longer than `max_sentence_len`
            word_list = word_list[:self.CONFIG.MAX_SENTENCE_LEN]
            word_list[-1] = '<EOS>'
        else:
            word_list = word_list + ['<PAD>'] * (self.CONFIG.MAX_SENTENCE_LEN - len(word_list))
        for word in word_list:
            index_list.append(self.word2idx[word])
        text_array = np.array(index_list)
        text_tensor = torch.from_numpy(text_array).type(torch.LongTensor)
        imgseq_tensor = torch.from_numpy(self.data[idx][1]).type(torch.FloatTensor)

        return text_tensor, imgseq_tensor, self.data[idx][2]


def transform_idx2word(index, idx2word):
    return " ".join([idx2word[str(idx)] for idx in index])


def transform_inverse_normalize(image_tensor):
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
    inv_tensor = inv_normalize(image_tensor)
    return inv_tensor


def save_models(checkpoint, path, prefix):
    if not os.path.isdir(path):
        os.makedirs(path)
    checkpoint_save_path = '{}/{}_epoch_{}.pt'.format(path, prefix, checkpoint['epoch'])
    torch.save(checkpoint, checkpoint_save_path)


def weights_xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)


class Dataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand < frac] = 0
    return data_noise


def align_cluster(label, cluster_id):
    label = np.array(label)
    cluster_id = np.array(cluster_id)
    assert label.size == cluster_id.size
    D = max(label.max(), cluster_id.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(label.size):
        w[label[i], cluster_id[i]] += 1
    print(pd.DataFrame(data=w, index=range(D), columns=range(D)))
    from scipy.optimize import linear_sum_assignment
    label_ind, cluster_ind = linear_sum_assignment(w.max() - w)
    print(label_ind)
    print(cluster_ind)
    return label_ind, cluster_ind


def count_percentage(cluster_labels):
    count = dict(collections.Counter(cluster_labels))
    sorted_count = sorted(count.items(), key=lambda x: x[0], reverse=False)
    for cluster in sorted_count:
        print("cluster {} : {:.2%}".format(str(cluster[0]), cluster[1] / len(cluster_labels)))


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)