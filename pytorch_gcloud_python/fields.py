import torchtext
from torchtext import data

# tokenizer function using spacy
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])


def tokenizer(s):
    return [w.text.lower() for w in nlp(tweet_clean(s))]


def tweet_clean(text):
    # remove non alphanumeric character
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = re.sub(r'https?:/\/\S+', ' ', text)  # remove links
    return text.strip()


# define the columns that we want to process and how to process
txt_field = data.Field(sequential=True,
                       tokenize=tokenizer,
                       include_lengths=True,
                       use_vocab=True)
label_field = data.Field(sequential=False,
                         use_vocab=False,
                         pad_token=None,
                         unk_token=None)

train_val_fields = [
    ('ItemID', None),  # we dont need this, so no processing
    ('Sentiment', label_field),  # process it as label
    ('SentimentSource', None),  # we dont need this, so no processing
    ('SentimentText', txt_field)  # process it as text
]

trainds, valds = data.TabularDataset.splits(path='./data',
                                            format='csv',
                                            train='traindf.csv',
                                            validation='valdf.csv',
                                            fields=train_val_fields,
                                            skip_header=True)

