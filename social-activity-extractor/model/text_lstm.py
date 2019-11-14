import datetime
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import normalized_mutual_info_score, f1_score, accuracy_score
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F


class LSTMClassifier(nn.Module):
    def __init__(self, device, batch_size, output_size, hidden_size, embedding, dropout):
        super(LSTMClassifier, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : Size of output activity classes
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained word_embeddings which we will use to create our word_embedding look-up table 

        """
        self.device = device
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.word_embeddings = embedding
        self.lstm0 = nn.LSTM(embedding.embedding_dim, hidden_size[0], dropout=dropout)
        self.dropout0 = nn.Dropout(p=dropout)
        self.lstm1 = nn.LSTM(hidden_size[0], hidden_size[1], dropout=dropout)
        self.dropout1 = nn.Dropout(p=dropout)
        self.lstm2 = nn.LSTM(hidden_size[1], hidden_size[2], dropout=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.label = nn.Linear(hidden_size[2], output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_sentence):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)

        """

        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
        input = self.word_embeddings(
            input_sentence)  # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        input = input.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        output0, _ = self.lstm0(input)
        output1, _ = self.lstm1(self.dropout0(output0))
        output2, (final_hidden_state, final_cell_state) = self.lstm2(self.dropout1(output1))
        final_output = self.label(self.dropout2(final_hidden_state[-1]))  # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
        log_prob = self.softmax(final_output)
        return log_prob

class TextModel(nn.Module):
    def __init__(self, device, text_encoder):
        super(self.__class__, self).__init__()
        self.device = device
        self.text_encoder = text_encoder
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

    def forward(self, x):
        log_prob = self.text_encoder(x)
        return log_prob

    def fit(self, train_dataset, lr=0.001, batch_size=256, num_epochs=10, save_path=None, tol=1e-3):
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        criterion = nn.NLLLoss().to(self.device)
        self.to(self.device)

        for epoch in range(num_epochs):
            self.train()
            # train 1 epoch
            train_loss = 0.0
            train_pred = []
            train_labels = []
            for batch_idx, input_batch in enumerate(trainloader):
                feature_batch = Variable(input_batch[1]).to(self.device)
                target_batch = Variable(input_batch[2]).to(self.device)
                optimizer.zero_grad()
                log_prob = self.forward(feature_batch)
                loss = criterion(log_prob, target_batch)
                loss.backward()
                optimizer.step()
                train_loss = train_loss + loss.data
                pred_batch = torch.argmax(log_prob, dim=1).cpu().numpy()
                train_pred.extend(pred_batch)
                train_labels.extend(target_batch.cpu().numpy())
                del feature_batch, target_batch, log_prob, loss
            train_loss = train_loss / len(trainloader)
            train_acc = accuracy_score(train_labels, train_pred)
            train_nmi = normalized_mutual_info_score(train_labels, train_pred)
            train_f_1 = f1_score(train_labels, train_pred)
            print("#Epoch %3d: acc: %.4f, nmi: %5f, f_1: %4f, loss: %.4f at %s" % (
                epoch + 1, train_acc, train_nmi, train_f_1, train_loss, str(datetime.datetime.now())))
        if save_path:
            self.save_model(os.path.join(save_path, "text_lstm.pt"))

    def predict(self, test_dataset, batch_size=256):
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False)
        self.to(self.device)
        self.eval()
        test_pred = []
        test_labels = []
        for batch_idx, input_batch in enumerate(testloader):
            feature_batch = Variable(input_batch[1]).to(self.device)
            target_batch = Variable(input_batch[2]).to(self.device)
            log_prob = self.forward(feature_batch)
            pred_batch = torch.argmax(log_prob, dim=1).cpu().numpy()
            test_pred.extend(pred_batch)
            test_labels.extend(target_batch.cpu().numpy())
            del feature_batch, target_batch, log_prob

        test_acc = accuracy_score(test_labels, test_pred)
        test_nmi = normalized_mutual_info_score(test_labels, test_pred)
        test_f_1 = f1_score(test_labels, test_pred)
        print("#Test acc: %.4f, Test nmi: %5f, Test f_1: %4f" % (
            test_acc, test_nmi, test_f_1))
        self.acc = test_acc
        self.nmi = test_nmi
        self.f_1 = test_f_1
