# ER
## Requirements
* Pyton 3.6
* Keras 2.2.4
* thesaurus
* SPARQLWrapper
* textdistance

## Pre-processing
Download ER_main.csv, ER_train.csv, ER_test.csv from here (https://drive.google.com/open?id=1RDreKI4osOWcXm4cOSfK-avxZonqGNEZ) and copy it inside "data/" directory.

For embedding download glove.6B.300d.txt (822MB) from https://nlp.stanford.edu/projects/glove/ and copy it inside "embedding/" directory.

## Models

#### Entity-Relation Predictor
You can use pre-trained model ER_model.json, ER_weight.h5 for Entity-Relation. For that download ER_weight.h5 (1.95GB) file from here (https://www.dropbox.com/s/qwk93abdf4whjux/ER_wights.h5?dl=0) and copy it inside /models/checkpoints/ directory.
For testing your custom input download 'word_to_id.json' file from here (https://www.dropbox.com/s/uidgm6zbbdw5gy6/word_to_id.json?dl=0) and copy it inside /models/checkpoints/ directory.

## Performance
