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
test_input = ["licensor","teleserial","james william whilt","recentemente","sugar sweetened", "Rivers", "writes"]
#                 R          R                  E              R               R               R        R

'''
for word in test_input:
    found = True
    tmp = []
    for w in word.split(" "):
        if w in word_to_id:
            tmp.append(word_to_id[w])
        else:
            found = False

    if found==False:
        continue
    tmp_padded = sequence.pad_sequences([tmp], maxlen=max_text_length)
    #loaded_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    prediction = loaded_model.predict(np.array([tmp_padded][0]))[0][0]
    if prediction<0.5:
        print(word+" -> Relation:{}%%".format(1-prediction))
    else:
        print(word+" -> Entity:{}%%".format(prediction))

    #print(loaded_model.predict_classes(np.array([tmp_padded][0])))
'''


def test_lcquad():

    prediction_stat = []
    threshold = 0.9975
    accuracy = 0.0
    with open('../data/Test_LC_QuAD.csv',encoding="utf-8") as f:
        rows = csv.reader(f, delimiter=',')
        correct = 0
        count = 0
        write_to_file = []
        for er in rows:
            found = True
            tmp = []

            er_word = er[0]
            er_type = er[1]

            for w in er_word.split(" "):
                if w in word_to_id:
                    tmp.append(word_to_id[w])
                else:
                    found = False

            if found == False:
                continue

            pred = ""
            prob = 0.0

            tmp_padded = sequence.pad_sequences([tmp], maxlen=max_text_length)
            prediction = loaded_model.predict(np.array([tmp_padded][0]))[0][0]

            if prediction < threshold:
                #print(word + " -> Relation:{}%%".format(1 - prediction))
                prob = (1-prediction)
                pred ="R"
            else:
                #print(word + " -> Entity:{}%%".format(prediction))
                prob= prediction
                pred = "E"

            dict = {
                "ER_word": er_word,
                "actual_type": er_type,
                "predicted_type": pred,
                "probability_of_being_entity": prob
            }
            prediction_stat.append(dict)
            write_to_file.append([er_word,er_type,pred,prob])   # Writing these information to the file

            if er_type==pred:
                correct+=1
            count+=1

            accuracy = (correct/count)*100

        to_write = open("../data/threshold__"+str(threshold)+"acc__"+str(accuracy)+".csv", "w")
        data_writer = csv.writer(to_write)
        data_writer.writerows(write_to_file)
        print( accuracy)



test_lcquad()




