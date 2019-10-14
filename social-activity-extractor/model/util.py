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


def load_multi_csv_data(args, CONFIG):
    df_image_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.image_csv), index_col=0,
                                encoding='utf-8-sig')
    scaled_image_data = MinMaxScaler((0, 5)).fit_transform(np.array(df_image_data.values))
    df_scaled_image_data = pd.DataFrame(data=scaled_image_data, index=df_image_data.index,
                                        columns=df_image_data.columns)
    df_text_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.text_csv), index_col=0,
                               encoding='utf-8-sig')
    scaled_text_data = MinMaxScaler((0, 5)).fit_transform(np.array(df_text_data.values))
    df_scaled_text_data = pd.DataFrame(data=scaled_text_data, index=df_text_data.index, columns=df_text_data.columns)
    short_codes = []
    image_data = []
    text_data = []
    pbar = tqdm(total=df_scaled_image_data.shape[0])
    for index, row in df_scaled_image_data.iterrows():
        short_codes.append(index)
        image_data.append(np.array(row))
        text_data.append(np.array(df_scaled_text_data.loc[index]))
        pbar.update(1)
    pbar.close()
    full_dataset = MultiCSVDataset(short_codes, np.array(image_data), np.array(text_data), CONFIG)
    return full_dataset


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


def load_csv_data(args, CONFIG):
    df_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.target_csv), index_col=0,
                          encoding='utf-8-sig')
    scaled_data = StandardScaler().fit_transform(np.array(df_data.values))
    df_scaled = pd.DataFrame(data=scaled_data, index=df_data.index, columns=df_data.columns)
    df_train, df_val = train_test_split(df_scaled, test_size=args.split_rate)
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


def load_text_data(args, CONFIG, word2idx):
    full_data = []
    df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'posts.csv'), header=None,
                          encoding='utf-8-sig')
    pbar = tqdm(total=df_data.shape[0])
    for index, row in df_data.iterrows():
        pbar.update(1)
        text_data = row.iloc[1]
        full_data.append(text_data)
        del text_data
    pbar.close()
    train_size = int(args.split_rate * len(full_data))
    val_size = len(full_data) - train_size
    train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])
    train_dataset, val_dataset = TextDataset(train_data, CONFIG, word2idx), \
                                 TextDataset(val_data, CONFIG, word2idx)
    return train_dataset, val_dataset


class TextDataset(Dataset):
    def __init__(self, data_list, CONFIG, word2idx):
        self.data = data_list
        self.word2idx = word2idx
        self.CONFIG = CONFIG

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word_list = self.data[idx].split()
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
        return text_tensor


def load_text_data_with_short_code(args, CONFIG, word2idx):
    full_data = []
    df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'posts.csv'), header=None,
                          encoding='utf-8-sig')
    pbar = tqdm(total=df_data.shape[0])
    for index, row in df_data.iterrows():
        pbar.update(1)
        short_code = row.iloc[0]
        text_data = row.iloc[1]
        full_data.append([text_data, short_code])
        del text_data
    pbar.close()
    full_dataset = TextDataset_with_short_code(full_data, CONFIG, word2idx)
    return full_dataset


class TextDataset_with_short_code(Dataset):
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
        return text_tensor, self.data[idx][1]


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
        image_path = row.iloc[2]
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


def align_cluster(image_cluster, text_cluster):
    assert image_cluster.size == text_cluster.size
    D = max(image_cluster.max(), text_cluster.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(image_cluster.size):
        w[image_cluster[i], text_cluster[i]] += 1
    print(pd.DataFrame(data=w, index=range(D), columns=range(D)))
    from scipy.optimize import linear_sum_assignment
    image_ind, text_ind = linear_sum_assignment(w.max() - w)
    return image_ind, text_ind


def count_percentage(cluster_labels):
    count = dict(collections.Counter(cluster_labels))
    sorted_count = sorted(count.items(), key=lambda x: x[0], reverse=False)
    for cluster in sorted_count:
        print("cluster {} : {:.2%}".format(str(cluster[0]), cluster[1] / len(cluster_labels)))
