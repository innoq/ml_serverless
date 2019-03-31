import datetime
import dill
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

from mllib import mllib

logging.basicConfig(level=logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# tokenizer function using spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])

logging.info(f'loaded nlp {nlp.meta["lang"]} {nlp.meta["name"]}')

tokenizer = mllib.create_tokenizer(nlp)

# define the columns that we want to process and how to process
txt_field = data.Field(sequential=True,
                       tokenize=tokenizer,
                       include_lengths=True,
                       use_vocab=True)
label_field = data.Field(sequential=False,
                         use_vocab=False,
                         pad_token=None,
                         unk_token=None)

logging.info(
    f'txt_field: {txt_field.dtype} / label_field: {label_field.dtype}')

train_val_fields = [
    ('ItemID', None),  # we dont need this, so no processing
    ('Sentiment', label_field),  # process it as label
    ('SentimentSource', None),  # we dont need this, so no processing
    ('SentimentText', txt_field)  # process it as text
]

logging.debug(f'{datetime.datetime.now()}    starting splits')
trainds, valds = data.TabularDataset.splits(path='./data',
                                            format='csv',
                                            train='traindf.csv',
                                            validation='valdf.csv',
                                            fields=train_val_fields,
                                            skip_header=True)

logging.info(f'trainds: {type(trainds)}')


logging.debug(f'{datetime.datetime.now()}    starting loading vectors')
# specify the path to the localy saved vectors
vec = vocab.Vectors('glove.twitter.27B.100d.txt', './data/glove.twitter.27B/')

logging.debug(
    f'{datetime.datetime.now()}    starting to build vocab on txt_field')
# build the vocabulary using train and validation dataset and assign the vectors
txt_field.build_vocab(trainds, valds, max_size=100000, vectors=vec)

logging.debug(
    f'{datetime.datetime.now()}    starting to build vocab on label_field')
# build vocab for labels
label_field.build_vocab(trainds)

traindl, valdl = data.BucketIterator.splits(datasets=(trainds, valds),  # specify train and validation Tabulardataset
                                            # batch size of train and validation
                                            batch_sizes=(3, 3),
                                            # on what attribute the text should be sorted
                                            sort_key=lambda x: len(
                                                x.SentimentText),
                                            device=None,  # -1 mean cpu and 0 or None mean gpu
                                            sort_within_batch=True,
                                            repeat=False)


train_batch_it = mllib.BatchGenerator(traindl, 'SentimentText', 'Sentiment')
val_batch_it = mllib.BatchGenerator(valdl, 'SentimentText', 'Sentiment')

vocab_size = len(txt_field.vocab)
embedding_dim = 100
n_hidden = 64
n_out = 2

# Training


def fit(model, train_dl, val_dl, loss_fn, opt, epochs=3):
    num_batch = len(train_dl)
    for epoch in trange(epochs):
        y_true_train = list()
        y_pred_train = list()
        total_loss_train = 0

        logging.debug(f'{datetime.datetime.now()}    starting epoch {epoch}')
        t = tqdm(iter(train_dl), leave=False, total=num_batch)
        for (X, lengths), y in t:
            t.set_description(f'Epoch {epoch}')
            lengths = lengths.cpu().numpy()

            opt.zero_grad()
            pred = model(X, lengths)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

            t.set_postfix(loss=loss.item())
            pred_idx = torch.max(pred, dim=1)[1]

            y_true_train += list(y.cpu().data.numpy())
            y_pred_train += list(pred_idx.cpu().data.numpy())
            total_loss_train += loss.item()

        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_loss = total_loss_train/len(train_dl)

        if val_dl:
            y_true_val = list()
            y_pred_val = list()
            total_loss_val = 0
            for (X, lengths), y in tqdm(val_dl, leave=False):
                pred = model(X, lengths.cpu().numpy())
                loss = loss_fn(pred, y)
                pred_idx = torch.max(pred, 1)[1]
                y_true_val += list(y.cpu().data.numpy())
                y_pred_val += list(pred_idx.cpu().data.numpy())
                total_loss_val += loss.item()
            valacc = accuracy_score(y_true_val, y_pred_val)
            valloss = total_loss_val/len(valdl)
            print(
                f'Epoch {epoch}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {valloss:.4f} val_acc: {valacc:.4f}')
        else:
            print(
                f'Epoch {epoch}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f}')


m = mllib.SimpleGRU(vocab_size, embedding_dim, n_hidden, n_out,
                    trainds.fields['SentimentText'].vocab.vectors, device=device).to(device)
opt = optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), 1e-3)

fit(model=m, train_dl=train_batch_it, val_dl=val_batch_it,
    loss_fn=F.nll_loss, opt=opt, epochs=2)

torch.save(m.state_dict(), './data/model.pt')
torch.save(txt_field, "./data/txt_field.pt", pickle_module=dill)
torch.save(label_field, "./data/label_field.pt", pickle_module=dill)
