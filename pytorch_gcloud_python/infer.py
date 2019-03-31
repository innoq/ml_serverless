import logging
import re

import spacy
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchtext
from torchtext import data
from torchtext import vocab
from tqdm import tqdm, trange

logging.basicConfig(level=logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# tokenizer function using spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])

logging.info(f'loaded nlp {nlp.meta}')


def tokenizer(s):
    return [w.text.lower() for w in nlp(tweet_clean(s))]


def tweet_clean(text):
    # remove non alphanumeric character
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = re.sub(r'https?:/\/\S+', ' ', text)  # remove links
    return text.strip()


torch.save(txt_field, "./data/txt_field.pt")
torch.save(label_field, "./data/label_field.pt", pickle_module=dill)

vocab_size = len(txt_field.vocab)
embedding_dim = 100
n_hidden = 64
n_out = 2


m = SimpleGRU(vocab_size, embedding_dim, n_hidden, n_out,
              trainds.fields['SentimentText'].vocab.vectors).to(device)
