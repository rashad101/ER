<<<<<<< HEAD
import numpy as np
import random
import csv

random.seed(101)

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

        train_d = open("ER_train.csv","w")
        data_w = csv.writer(train_d)
        data_w.writerows(train_data)

        test_d = open("ER_test.csv", "w")
        data_w = csv.writer(test_d)
        data_w.writerows(test_data)

split_data()
