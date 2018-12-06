import json
import csv
import random
import numpy as np
from keras.models import  model_from_json
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

random.seed(101)

data = []
with open('checkpoints/word_to_id.json') as f:
    data = json.load(f)
word_to_id = data[0]

json_file = open('checkpoints/ER_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("checkpoints/ER_wights.h5")
print("Loaded model from disk")


max_text_length = 500
test_input = ["licensor","teleserial","james william whilt"]
#                 R          R                  E



for word in test_input:
    tmp = []
    for w in word.split(" "):
        tmp.append(word_to_id[w])
    tmp_padded = sequence.pad_sequences([tmp], maxlen=max_text_length)
    #loaded_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    prediction = loaded_model.predict(np.array([tmp_padded][0]))[0][0]
    if prediction<0.5:
        print(word+" -> Relation:{}%%".format(1-prediction))
    else:
        print(word+" -> Entity:{}%%".format(prediction))

    #print(loaded_model.predict_classes(np.array([tmp_padded][0])))
