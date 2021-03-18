# import eland as ed
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# from elasticsearch import Elasticsearch
#
# # Connecting to an Elasticsearch instance running on 'localhost:9200'
# ed_data = ed.DataFrame("localhost:9200", "index_mongo")
#
# # conversion from eland data type to pandas data type
# pd_data = ed.eland_to_pandas(ed_data)
# print(type(pd_data))
# print(pd_data.tail())
#
# # print columns/features of data
# print(pd_data.columns)
#
# # select only columns that have values of type number
# print(pd_data.select_dtypes(include=np.number))
#
# # select only the columns we want
# print(pd_data.get(['ip', 'sessionID', 'paid']).tail())
#
# print("--------------")
# output_form = pd_data.groupby('sessionID')[['ip','ips','paid','method','statusCode','userEmail',]].apply(lambda x: x.values.tolist())
# print(output_form)
# # output_form.to_csv('out.scv')
#
# print(len(output_form))
#
# selected_output = pd.Series([])
# i = 0
# for index, row in output_form.iteritems():
#     if(len(row)) >= 5:
#         selected_output[i] = row
#         i = i + 1
#
# print(selected_output)
# selected_output.to_csv('selected_out.csv')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymongo
pd.options.mode.chained_assignment = None  # default='warn'
import six
import sys
sys.modules['sklearn.externals.six'] = six
from seqlearn.datasets import load_conll
from seqlearn.perceptron import StructuredPerceptron
from seqlearn.evaluation import bio_f_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from seqlearn.hmm import MultinomialHMM
# from nose.tools import assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal
from keras.preprocessing import sequence
from sequence_classifiers import CNNSequenceClassifier
from keras.datasets import imdb
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix





myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["amazona"]
mycol = mydb["datas"]


data = pd.DataFrame(list(mycol.find()))
print(data.columns)

BenignData = []
BenignData = data[data.timestamp > 1608634273]
print(BenignData)

# BenignData = BenignData.replace(",", "")
BenignData.to_csv('BenignData.csv')


# print(BenignData["baseUrl"])
# print(BenignData["path"])
#
#
# print(BenignData["baseUrl"] + BenignData["path"])
BenignData["state"] = BenignData["baseUrl"] + BenignData["path"]
print(BenignData["state"])

# BenignDataTransactionsState1 = (BenignData.groupby('sessionID_alekos')['state'].apply(lambda x: x.values.tolist())).reset_index()
# print(BenignDataTransactionsState1)
# BenignDataTransactionsState1.to_csv('BenignDataTransactionsState1.csv')
# print(type(BenignDataTransactionsState1))

BenignDataTransactionsState = BenignData.groupby('sessionID_alekos')['state'].agg(' '.join).reset_index()
print(BenignDataTransactionsState)
BenignDataTransactionsState.to_csv('BenignDataTransactionsState.csv')
print(type(BenignDataTransactionsState))


BenignDataTransactionsStateFinal = BenignDataTransactionsState.drop('sessionID_alekos', axis=1)
BenignDataTransactionsStateFinal.insert(1, 'Label', '0')
BenignDataTransactionsStateFinal['Final'] = BenignDataTransactionsStateFinal['state'].str.cat(BenignDataTransactionsStateFinal['Label'], sep=" ")
BenignDataTransactionsStateFinal = BenignDataTransactionsStateFinal.drop('state', axis=1)
BenignDataTransactionsStateFinal = BenignDataTransactionsStateFinal.drop('Label', axis=1)

# zeros = np.where(np.empty_like(BenignDataTransactionsStateFinal.values), "", "")
# data = np.hstack([BenignDataTransactionsStateFinal.values, zeros]).reshape(-1, BenignDataTransactionsStateFinal.shape[1])
# BenignDataTransactionsStateFinal = pd.DataFrame(data, columns=BenignDataTransactionsStateFinal.columns)

print(BenignDataTransactionsStateFinal)
BenignDataTransactionsStateFinal.to_csv('BenignDataTransactionsStateFinal.csv', header=False, index=False)


# Machine Learning

# Checks for the largest common prefix
def lcp(s, t):
    n = min(len(s), len(t))
    for i in range(0, n):
        if s[i] != t[i]:
            return s[0:i]
    else:
        return s[0:n]


def repeats(seq):
    lrs = ""
    n = len(seq)
    for i in range(0, n):
        for j in range(i + 1, n):
            # Checks for the largest common factors in every substring
            x = lcp(seq[i:n], seq[j:n])
            # If the current prefix is greater than previous one
            # then it takes the current one as longest repeating sequence
            if len(x) > len(lrs):
                lrs = x
    # print("Longest repeating sequence: " + lrs)
    return lrs


def features(sequence, i):
    if sequence[i]:
        yield str(sequence[i].count('pay'))
        yield str(sequence[i].count('products'))
        yield str(sequence[i].count('signin'))
        # yield str(sequence[i].count('reviews'))
        yield repeats(sequence[i])

def featuresHMM(sequence, i):
    if sequence[i]:
        # print(sequence[i].count('pay'))
        # yield str(sequence[i].count('pay'))
        yield str(sequence[i].count('products'))
        # yield str(sequence[i].count('signin'))
        # yield str(sequence[i].count('reviews'))
        # print("repeats")
        # yield repeats(sequence[i])


def featuresVerbResponse(sequence, i):
    if sequence[i]:
        yield str(sequence[i].count('pay'))
        yield str(sequence[i].count('products'))
        yield str(sequence[i].count('signin'))
        yield repeats(sequence[i])
        yield str(sequence[i].count('200'))
        yield str(sequence[i].count('401'))
        yield str(sequence[i].count('PUT'))
        yield str(sequence[i].count('POST'))


print("---Load Data (Only pages as States)---")
X_train, y_train, lengths_train = load_conll("TrainDataPages.txt", features)
print("X_train")
print(X_train)
print("Y_train")
print(y_train)
print("lengths_trains")
print(lengths_train)

print("---Train Model---")

clf = StructuredPerceptron()
clf.fit(X_train, y_train, lengths_train)

print("---Validation Test---")

X_test, y_test, lengths_test = load_conll("TestDataPages.txt", features)
y_pred = clf.predict(X_test, lengths_test)
print("SCORE")
print("y_test")
print(y_test)
print("y_pred")
print(y_pred)
print(bio_f_score(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {0:0.2f}'.format(accuracy))
precision = precision_score(y_test, y_pred, average="binary", pos_label="0")
print('Precision: {0:0.2f}'.format(precision))
recall_average = recall_score(y_test, y_pred, average="binary", pos_label="0")
print('Recall: {0:0.2f}'.format(recall_average))
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(clf, X_test, y_test, cmap='Blues')
plt.show()


print("--------------------")

print("---Load Data (PagesVerbsResponse as States)---")
X_train, y_train, lengths_train = load_conll("TrainDataPagesVerbResponse.txt", featuresVerbResponse)
print("X_train")
print(X_train)
print("Y_train")
print(y_train)
print("lengths_trains")
print(lengths_train)

print("---Train Model---")

clf = StructuredPerceptron()
clf.fit(X_train, y_train, lengths_train)

print("---Validation Test---")

X_test, y_test, lengths_test = load_conll("TestDataPagesVerbResponse.txt", features)
y_pred = clf.predict(X_test, lengths_test)
print("SCORE")
print("y_test")
print(y_test)
print("y_pred")
print(y_pred)
print(bio_f_score(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {0:0.2f}'.format(accuracy))
precision = precision_score(y_test, y_pred, average="binary", pos_label="0")
print('Precision: {0:0.2f}'.format(precision))
recall_average = recall_score(y_test, y_pred, average="binary", pos_label="0")
print('Recall: {0:0.2f}'.format(recall_average))
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(clf, X_test, y_test, cmap='Blues')
plt.show()

print("---Markov HMM---")
print("---Load Data (Pages as States)---")
X_train, y_train, lengths_train = load_conll("TrainDataPages.txt", featuresHMM)
X_train_ALL, y_train_ALL, lengths_train_ALL = load_conll("AllDataPages.txt", featuresHMM)
X_test, y_test, lengths_test = load_conll("TestDataPages.txt", featuresHMM)


print("X_train")
print(X_train)
print(X_train.nonzero()[1])

# hot encode X_train
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(X_train.nonzero()[1])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
X_train_encoded = onehot_encoder.fit_transform(integer_encoded)


# hot encode X_train_ALL
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(X_train_ALL.nonzero()[1])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
X_train_ALL_encoded = onehot_encoder.fit_transform(integer_encoded)


test_data_hmm = X_train_ALL_encoded[1035:1134, :]

print("X_train_encoded")
print(X_train_encoded)
print("X_train_encoded length")
print(X_train_encoded.shape)

print("Y_train")
print(y_train)
print("Y_train length")
print(y_train.shape)
print("lengths_trains")
print(lengths_train)

print("---Train Model---")

hmm = MultinomialHMM(decode='viterbi', alpha=0.01)
hmm.fit(X_train_encoded, y_train, lengths_train)

print("---Validation Test---")

y_pred = hmm.predict(test_data_hmm, lengths_test)
print("SCORE")
print("y_test")
print(y_test)
print("y_pred")
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {0:0.2f}'.format(accuracy))
precision = precision_score(y_test, y_pred, pos_label="0")
print('Precision: {0:0.2f}'.format(precision))
recall_average = recall_score(y_test, y_pred, average="binary", pos_label="0")
print('Recall: {0:0.2f}'.format(recall_average))
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(hmm, test_data_hmm, y_test, cmap='Blues')
plt.show()

print("---Load Data (PagesVerbResponse as States)---")
X_train, y_train, lengths_train = load_conll("TrainDataPagesVerbResponse.txt", featuresHMM)
X_train_ALL, y_train_ALL, lengths_train_ALL = load_conll("AllDataPagesVerbResponse.txt", featuresHMM)
X_test, y_test, lengths_test = load_conll("TestDataPagesVerbResponse.txt", featuresHMM)


print("X_train")
print(X_train)
print(X_train.nonzero()[1])

# hot encode X_train
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(X_train.nonzero()[1])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
X_train_encoded = onehot_encoder.fit_transform(integer_encoded)

# hot encode X_test
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(X_test.nonzero()[1])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
X_test_encoded = onehot_encoder.fit_transform(integer_encoded)


# hot encode X_train_ALL
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(X_train_ALL.nonzero()[1])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
X_train_ALL_encoded = onehot_encoder.fit_transform(integer_encoded)

train_data_hmm = X_train_ALL_encoded[0:1035, :]
test_data_hmm = X_train_ALL_encoded[1035:1134, :]



print("Y_train")
print(y_train)
print("lengths_trains")
print(lengths_train)
print("test_data_hmm")
print(test_data_hmm.shape)
print("lengths_test")
print(lengths_test)
print("X_train_encoded")
print(X_train_encoded.shape)
print("train_data_hmm")
print(train_data_hmm.shape)
print("test_data_hmm length")
print(test_data_hmm.shape)
print("---Train Model---")

hmm = MultinomialHMM()
hmm.fit(train_data_hmm, y_train, lengths_train)

print("---Validation Test---")

print("X_test_encoded length")
print(X_test_encoded.shape)
print("lengths_test")
print(lengths_test)

y_pred = hmm.predict(test_data_hmm, lengths_test)
print("SCORE")
print("y_test")
print(y_test)
print("y_pred")
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {0:0.2f}'.format(accuracy))
precision = precision_score(y_test, y_pred, pos_label="0")
print('Precision: {0:0.2f}'.format(precision))
recall_average = recall_score(y_test, y_pred, average="binary", pos_label="0")
print('Recall: {0:0.2f}'.format(recall_average))
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(hmm, test_data_hmm, y_test, cmap='Blues')
plt.show()



# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(X_train, y_train)
# y_pred = neigh.predict(X_test)
# print(y_pred)
# print(y_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy: {0:0.2f}'.format(accuracy))
