import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import LSTM,Bidirectional, Dense, Embedding, Activation
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import os, re, csv, math, codecs

'''
rows=[["data","type"]]
rows.append(['Rony',0])
rows.append(['capital of',1])
rows.append(['Barack Obama',0])
rows.append(['president',1])
rows.append(['Germany',0])
rows.append(['vice president',1])
rows.append(['USA',0])
rows.append(['biggest',1])
rows.append(['Berlin',0])


file = open("data/train.csv","w", encoding="utf-8")
w = csv.writer(file)
wr = w.writerows(rows)
file.close()
'''




rows=[["data","type"]]
rows.append(['robert',"E"])
rows.append(['district of',"R"])
rows.append(['Barack Obama',"E"])
rows.append(['railway',"R"])
rows.append(['largest river',"R"])


file = open("data/test.csv","w", encoding="utf-8")
w = csv.writer(file)
wr = w.writerows(rows)
file.close()


DATA_PATH = 'data/'
EMBEDDING_DIR = 'data/fasttext'

MAX_NB_WORDS = 25
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

def normalize(s):
    s = str(s)
    s = s.lower()
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    return s

from subprocess import check_output
print(check_output(["ls", "data"]).decode("utf8"))



#load data
train_df = pd.read_csv('data/ER_main_v2.csv', sep=',', header=0)
test_df = pd.read_csv('data/lcquad_ER.csv', sep=',', header=0)
test_df = test_df.fillna('_NA_')

print("num train: ", train_df.shape[0])
print("num test: ", test_df.shape[0])

label_names = ["E","R"]
y_train = train_df["type"].values
y_train = [0 if r=="E" else 1 for r in y_train]
#print(y_train)
y_test = test_df["type"].values
y_test = [0 if r=="E" else 1 for r in y_test]
train_df['data'] = [normalize(w) for w in train_df['data']]
print(y_test)



#visualize word distribution
train_df['doc_len'] = train_df['data'].apply(lambda words: len(words.split(" ")))
max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)
sns.distplot(train_df['doc_len'], hist=True, kde=True, color='b', label='doc len')
plt.axvline(x=max_seq_len, color='k', linestyle='--', label='max len')
plt.title('data length'); plt.legend()
#plt.show()



raw_docs_train = train_df['data'].tolist()
raw_docs_test = test_df['data'].tolist()
num_classes = len(label_names)

print("pre-processing train data...")
processed_docs_train = []
for doc in tqdm(raw_docs_train):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_train.append(" ".join(filtered))
#end for

processed_docs_test = []
for doc in tqdm(raw_docs_test):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_test.append(" ".join(filtered))
print(processed_docs_test)
#end for

print("tokenizing input data...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=True)
tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  #leaky
word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
word_index = tokenizer.word_index
print("dictionary size: ", len(word_index))
#print(word_index.keys())

#pad sequences
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)
print("word_eq_test",word_seq_test)
print(y_test)










#load embeddings
print('loading word embeddings...')
embeddings_index = {}
f = codecs.open('data/fasttext/wiki-news-300d-1M.vec', encoding='utf-8')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))


#training params
batch_size = 2
num_epochs = 50

#model parameters
embed_dim = 300


#embedding matrix
print('preparing embedding matrix...')
words_not_found = []
nb_words = min(MAX_NB_WORDS, len(word_index))

nb_words+=1
print("nb_words: ",nb_words)


embedding_matrix = np.zeros((nb_words, embed_dim))
for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


#print("sample words not found: ", np.random.choice(words_not_found, 10))



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



embedding_layer =Embedding(nb_words, embed_dim, weights=[embedding_matrix], input_length=max_seq_len, trainable=False)

# Defining model

model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(300,dropout=0.3, activation="relu")))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy',f1])
print(model.summary())


early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=7, verbose=1)
callbacks_list = [early_stopping]

hist = model.fit(word_seq_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.1, shuffle=True, verbose=2)


y_test = model.evaluate(word_seq_test,y_test)
print(word_seq_test)
print("y test",y_test)
text = "president of"



alist = ["robert","jhon","district of", "railway"]

for text in alist:

    seq = np.array(tokenizer.texts_to_sequences(text))
    seq2 = tokenizer.texts_to_sequences(text)

    #prediction = model.predict(np.array(sequence.pad_sequences([seq], maxlen=max_seq_len)))
    prediction2 = model.predict(sequence.pad_sequences(seq2, maxlen=max_seq_len))
    #print(prediction)
    print('prediction',prediction2)
    y_classes = prediction2.argmax(axis=-1)
    print(y_classes)
    print(prediction2.argmax())
    print(prediction2[0][0])
print(model.metrics_names)

