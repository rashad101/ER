import random
import csv
import pandas as pd
import keras
from numpy import array
import numpy as np
from keras.preprocessing.text import one_hot
from  keras.datasets import imdb
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import LSTM,Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


random.seed(101)

def f1(y_true, y_pred):

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        print("recall: {}".format(recall))
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    print("precision: {}".format(precision))
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



# load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r', encoding="utf-8")
	lines = file.readlines()
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = np.asarray(parts[1:],dtype="float32")
	return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = np.zeros((vocab_size, 300))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		vector = embedding.get(word)
		if vector is not None:
			weight_matrix[i] = vector
	return weight_matrix


def split_data():

    d = {}
    with open('../data/ER_main.csv', encoding="utf-8") as f:
        rows = csv.reader(f, delimiter=',')

        for row in rows:
            d[row[0]] = row[1]

        #shuffling the dictionary
        keys = list(d.keys())
        random.shuffle(keys)
        total_keys = len(keys)

        #initializing number of data for train and test
        n_train = round((70/100)*total_keys)
        n_test = total_keys-n_train

        train_data = []
        train_data.append(['data','type'])
        test_data = []
        test_data.append(['data','type'])

        i = 0
        for key in keys:
            if i<n_train:
                train_data.append([key,d[key]])
            else:
                test_data.append([key,d[key]])
            i += 1

        train_d = open("../data/ER_train.csv","w")
        data_w = csv.writer(train_d)
        data_w.writerows(train_data)

        test_d = open("../data/ER_test.csv", "w")
        data_w = csv.writer(test_d)
        data_w.writerows(test_data)


def load_data():

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    with open("../data/ER_train.csv", encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for row in csv_reader:
            if row[0]!='data':
                X_train.append(row[0])
                if row[1]=='E':
                    y_train.append(1)
                else:
                    y_train.append(0)
    f.close()
    print("Loaded {} train data.".format(len(X_train)))

    with open("../data/ER_test.csv", encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for row in csv_reader:
            if row[0]!='data':
                X_test.append(row[0])
                if row[1]=='E':
                    y_test.append(1)
                else:
                    y_test.append(0)
    f.close()
    print("Loaded {} test data.".format(len(X_test)))
    print("Data loading done !")

    return (X_train,y_train,X_test,y_test)

#split_data()

print("Loading data .....")
X_train, y_train,X_test, y_test = load_data()

print("Tokenizing text ......")
t = Tokenizer()
t.fit_on_texts(X_train+X_test)
print("Tokenizing Done.")
vocab_size = len(t.word_index)+1
print("vocab size: {}".format(vocab_size))



word_to_id = t.word_index

# truncate and pad the review sequences
max_text_length = 500
X_train = sequence.pad_sequences(t.texts_to_sequences(X_train), maxlen=max_text_length)
X_test = sequence.pad_sequences(t.texts_to_sequences(X_test), maxlen=max_text_length)


...
# load embedding from file
raw_embedding = load_embedding('../embedding/glove.6B.300d.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, word_to_id)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 300, weights=[embedding_vectors], input_length=max_text_length, trainable=False)


# create the model
model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(300)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy',f1])
print(model.summary())


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

# serialize model to JSON
model_json = model.to_json()
with open("checkpoints/ER_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("checkpoints/ER_wights.h5")
print("Saved model to disk")


print("Accuracy: %.2f%%" % (scores[1]*100))

entity = "Barack Obama"
relation = "president of"
for review in [entity,relation]:
    tmp = []
    for word in review.split(" "):
        tmp.append(word_to_id[word])
    tmp_padded = sequence.pad_sequences([tmp], maxlen=max_text_length)
    print("%s . Entity: %s" % (review,model.predict(array([tmp_padded][0]))[0][0]))



#For Testing with loaded model

'''
# load json and create model
json_file = open('checkpoints/ER_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


# load weights into new model
loaded_model.load_weights("checkpoints/ER_wights.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy',f1])
score = loaded_model.evaluate(X_test, y_test, verbose=0)

print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
'''