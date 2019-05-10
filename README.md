### Examples how you can serve machine learning models as cloud function

Right now we have 7 examples:

-----------------------------
- diabetes prediction (structured data), implemented with scikit-learn, running on Python 3.6 at Microsoft Azure Functions
- diabetes prediction (structured data), implemented with scikit-learn, running on Python 3.6 at Google Cloud Functions
- german tweets categorization (sentiment analysis), implemented with SpaCy, running on Python 3.6 at Google Cloud Functions
- german tweets categorization (sentiment analysis), implemented with SpaCy and Pytorch, running on Python 3.6 at Google Cloud Functions
- flower images categorization, implemented with Tensorflow, running on Python 2.7 at AWS Lambda
- flower images categorization, implemented with Tensorflow, running on Node 8 at Google Cloud Functions
- flower images categorization, implemented with Tensorflow, running on Node 8 at AWS Lambda (currently works in offline mode only)

Generally you have to follow API guidelines provided by AWS, GCloud or Azure. In cases you have to do something else or additional we point it out in README.

Feel free to contribute!

Licensed under the Apache License, Version 2.0