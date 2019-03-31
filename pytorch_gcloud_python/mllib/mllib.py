import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def create_tokenizer(nlp):
    def tokenizer(s):
        return [w.text.lower() for w in nlp(tweet_clean(s))]
    return tokenizer


def tweet_clean(text):
    # remove non alphanumeric character
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = re.sub(r'https?:/\/\S+', ' ', text)  # remove links
    return text.strip()


class BatchGenerator:
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (X, y)


class SimpleGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, n_out, pretrained_vec, bidirectional=True, device=None):
        super().__init__()
        self.vocab_size, self.embedding_dim, self.n_hidden, self.n_out, self.bidirectional = vocab_size, embedding_dim, n_hidden, n_out, bidirectional
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.emb.weight.data.copy_(pretrained_vec)
        self.emb.weight.requires_grad = False
        self.gru = nn.GRU(self.embedding_dim, self.n_hidden,
                          bidirectional=bidirectional)
        self.out = nn.Linear(self.n_hidden, self.n_out)
        self.device = device

    def forward(self, seq, lengths):
        bs = seq.size(1)  # batch size
        seq = seq.transpose(0, 1)
        self.h = self.init_hidden(bs)  # initialize hidden state of GRU
        embs = self.emb(seq)
        embs = embs.transpose(0, 1)
        embs = pack_padded_sequence(embs, lengths)  # unpad
        # gru returns hidden state of all timesteps as well as hidden state at last timestep
        gru_out, self.h = self.gru(embs, self.h)
        # pad the sequence to the max length in the batch
        gru_out, lengths = pad_packed_sequence(gru_out)
        # since it is as classification problem, we will grab the last hidden state
        # self.h[-1] contains hidden state of last timestep
        outp = self.out(self.h[-1])
#         return F.log_softmax(outp, dim=-1)
        return F.log_softmax(outp, dim=-1)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return torch.zeros((2, batch_size, self.n_hidden)).to(self.device)
        else:
            return torch.zeros((1, batch_size, self.n_hidden)).to(self.device)
