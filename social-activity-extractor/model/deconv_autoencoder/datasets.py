import torch
from torch.utils.data import Dataset

import os
import sys
import numpy as np
from tqdm import tqdm
from collections import Counter
from copy import deepcopy

def load_hotel_review_data(sentence_len):
    """
    Load Hotel Reviews data from pickle distributed in https://drive.google.com/file/d/0B52eYWrYWqIpQzhBNkVxaV9mMjQ/view
    This file is published in https://github.com/dreasysnail/textCNN_public
    
    :param path: pickle path
    :return: 
    """
    import _pickle as cPickle
    with open("data/pickle/hotel_reviews.p", "rb") as f:
        data = cPickle.load(f, encoding="latin1")

    train_data, test_data = HotelReviewsDataset(data[0], deepcopy(data[2]), deepcopy(data[3]), sentence_len, transform=ToTensor()), \
                             HotelReviewsDataset(data[1], deepcopy(data[2]), deepcopy(data[3]), sentence_len, transform=ToTensor())
    return train_data, test_data


class HotelReviewsDataset(Dataset):
    """
    Hotel Reviews Dataset
    """
    def __init__(self, data_list, word2index, index2word, sentence_len, transform=None):
        self.word2index = word2index
        self.index2word = index2word
        self.n_words = len(self.word2index)
        self.data = data_list
        self.sentence_len = sentence_len
        self.transform = transform
        self.word2index["<PAD>"] = self.n_words
        self.index2word[self.n_words] = "<PAD>"
        self.n_words += 1
        temp_list = []
        for sentence in tqdm(self.data):
            if len(sentence) > self.sentence_len:
                # truncate sentence if sentence length is longer than `sentence_len`
                temp_list.append(np.array(sentence[:self.sentence_len]))
            else:
                # pad sentence  with '<PAD>' token if sentence length is shorter than `sentence_len`
                sent_array = np.lib.pad(np.array(sentence),
                                        (0, self.sentence_len - len(sentence)),
                                        "constant",
                                        constant_values=(self.n_words-1, self.n_words-1))
                temp_list.append(sent_array)
        self.data = np.array(temp_list, dtype=np.int32)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform:
            data = self.transform(data)
        return data

    def vocab_lennght(self):
        return len(self.word2index)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, data):
        return torch.from_numpy(data).type(torch.LongTensor)
