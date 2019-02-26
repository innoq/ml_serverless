import spacy
import json
import logging

nlp = spacy.load('de_cat_100d_100k')

def spacy_sentiment(request):
    request_json = request.get_json()
    if request.args and 'message' in request.args:
        text = request.args.get('message')
    elif request_json and 'message' in request_json:
        text = request.get_json()['message']
    else:
        text = "Warum sind die Geier so gierig?"
    
    return encode_result(text, processText(text))

def encode_result(text, result, encoding='json'):
    return json.dumps([{'text': text, 'offense': result}])

def processText(mytext):
    global nlp
    doc = nlp(mytext)
    logging.info(f'RESULT: {doc.cats["OFFENSE"]:.4f} {mytext}')
    return doc.cats['OFFENSE']