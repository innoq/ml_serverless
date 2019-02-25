# Spacy Text Classifier on Google Cloud
The model used here does sentiment analysis for german text. It was trained with data of GermEval2018. Jupyter Notebook `train_spacy_model` shows all the steps for creating and training the model.

`main.py` is the program for inference, directory `de_cat_100d_100k` is the spacy model.

The model is beyond 100MB, so in case of AWS it would be needed to store it in a DataStore and to access it from there.

Function deployment with `gcloud functions deploy spacy_sentiment --memory=1024MB --runtime=python37 --trigger-http --source=.`