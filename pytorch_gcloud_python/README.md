## Torchtext sentiment analysis example

Using training data from http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip

Vectors from https://nlp.stanford.edu/data/glove.twitter.27B.zip

`python train.py` will create a preprocessing pipeline, vocab objects, a simple model, run the training (which could run longer) and store the resulting coefficients and mappings in 3 blob files

`main.py` contains the code run as gcloud function

Function deployment with `gcloud functions deploy torchtext_sentiment --memory=2048MB --runtime=python37 --trigger-http --source=.`