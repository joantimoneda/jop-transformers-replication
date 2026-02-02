
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Activation, Dropout, Dense, Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, precision_score, recall_score
import pandas as pd
import numpy as np
import string
import re
import os
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
APP_DIR = SCRIPT_DIR.parent
os.chdir(APP_DIR)

data = pd.read_excel("fake_news_covid.xlsx")
data = data[["label_text", "text"]]
dataF = data[data.label_text=="F"][0:500]
dataT = data[data.label_text=="T"][0:500]
dataU = data[data.label_text=="U"][0:500]
data = pd.concat([dataF, dataT, dataU])
len(data)
data["label_text"].value_counts()
data = data.sample(frac=1).reset_index(drop=True)
print(len(data))
data["text"] = data["text"].str.replace('\n', ' ')
data["label"] = data["label_text"].astype('category')
data["label"] = data["label"].cat.codes

data['text'].dropna(inplace=True)
data['text'] = data['text'].astype(str)
data['text'] = [entry.lower() for entry in data['text']]

def remove_tags(string):
    result = re.sub('<.*?>','',string)
    return result

data['text']= data['text'].apply(lambda cw : remove_tags(cw))
data.head()

## Tokenize 
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data["text"])
words_to_index = tokenizer.word_index
len(words_to_index) ## keeping all words

def read_glove_vector(glove_vec):
    with open(glove_vec, 'r', errors = 'ignore', encoding='utf8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            w_line = line.split(' ')
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
    return word_to_vec_map

word_to_vec_map = read_glove_vector('../Irwin twitter project/Baseline models/glove_vectors_w2v.txt')

X_all = tokenizer.texts_to_sequences(data["text"])
maxLen = max([len(x) for x in X_all])

vocab_len = len(words_to_index)
embed_vector_len = word_to_vec_map['moon'].shape[0]

emb_matrix = np.zeros((vocab_len+1, embed_vector_len))

for word, index in words_to_index.items():
    embedding_vector = word_to_vec_map.get(word)
    if embedding_vector is not None:
        emb_matrix[index, :] = embedding_vector

## Define general LSTM RNN MODEL:
def lstm_model(vocab_len, embed_vector_len):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_len, embed_vector_len, weights = [emb_matrix]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embed_vector_len)),
        tf.keras.layers.Dense(embed_vector_len, activation='gelu'),
        tf.keras.layers.Dense(3, activation='softmax')])
    return model 

## 10-fold Cross-Validation: 

ov_acc = []
f1 = []
acc = []
prec = []
rec = []

for x in range(0,10):
    kfold = KFold(n_splits=10, shuffle=True)    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
        print(f'FOLD {fold}')
        train = data.iloc[train_ids,]
        test = data.iloc[test_ids,]
        X_train = pd.Series(train["text"])
        Y_train = pd.Series(train["label"])
        X_test = pd.Series(test["text"])
        Y_test = pd.Series(test["label"])
        #Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
        #Y_test = np.asarray(Y_test).astype('float32').reshape((-1,1))
        
        ## TRAINING X indices
        X_train_indices = tokenizer.texts_to_sequences(X_train)
        X_train_indices = pad_sequences(X_train_indices, maxlen=maxLen, padding='post')
        X_train_indices.shape
        
        # Start model
        model = lstm_model(vocab_len+1, embed_vector_len)
        adam = tf.keras.optimizers.Adam(learning_rate = 0.0008)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        
        # Fit model
        model.fit(X_train_indices, Y_train, batch_size=64, epochs=5)

        # Test model
        X_test_indices = tokenizer.texts_to_sequences(X_test)
        X_test_indices = pad_sequences(X_test_indices, maxlen=maxLen, padding='post')
        model.evaluate(X_test_indices, Y_test)
        
        # Generate predictions
        preds = model.predict(X_test_indices)
        preds = pd.Series([np.argmax(x) for x in preds])
        preds.value_counts()

        ov_acc.append([accuracy_score(preds, Y_test), recall_score(preds, Y_test, average="macro"), precision_score(preds, Y_test, average="macro"),f1_score(preds, Y_test, average="macro")])
        f1.append(list(f1_score(Y_test,preds,average=None)))
        matrix = confusion_matrix(Y_test, preds)
        acc.append(list(matrix.diagonal()/matrix.sum(axis=1)))
        cr = pd.DataFrame(classification_report(Y_test,preds, output_dict=True)).transpose().iloc[0:3, 0:2]
        prec.append(list(cr.iloc[:,0]))
        rec.append(list(cr.iloc[:,1]))

splits = np.array_split(f1, 10)

mean_f1s = []
for x in range(0, len(splits)):
    mean_f1s.append(np.mean(splits[x][:,0]))
    mean_f1s.append(np.mean(splits[x][:,1]))
    mean_f1s.append(np.mean(splits[x][:,2]))

splits = np.array_split(mean_f1s, 10)

np.mean([x[0] for x in splits])
np.mean([x[1] for x in splits])
np.mean([x[2] for x in splits])

np.std([x[0] for x in splits])
np.std([x[1] for x in splits])
np.std([x[2] for x in splits])


lstm_glove_fakenews_stats = []
lstm_glove_fakenews_stats.append(
    {
        'Model': 'LSTM with GloVe (fakenews)',

        'fake_mean': np.mean([x[0] for x in acc]),
        'fake_mean_sd': np.std([x[0] for x in acc]),
        'fake_mean_f1': np.mean([x[0] for x in f1]),
        'fake_mean_f1_sd': np.std([x[0] for x in f1]),
        'fake_recall': np.mean([x[0] for x in rec]),
        'fake_recall_sd': np.std([x[0] for x in rec]),
        'fake_prec': np.mean([x[0] for x in prec]),
        'fake_prec_sd': np.std([x[0] for x in prec]),
        
        'true_mean': np.mean([x[1] for x in acc]),
        'true_mean_sd': np.std([x[1] for x in acc]),
        'true_mean_f1': np.mean([x[1] for x in f1]),
        'true_mean_f1_sd': np.std([x[1] for x in f1]),
        'true_recall': np.mean([x[1] for x in rec]),
        'true_recall_sd': np.std([x[1] for x in rec]),
        'true_prec': np.mean([x[1] for x in prec]),
        'true_prec_sd': np.std([x[1] for x in prec]),
        
        'undet_mean': np.mean([x[2] for x in acc]),
        'undet_mean_sd': np.std([x[2] for x in acc]),
        'undet_mean_f1': np.mean([x[2] for x in f1]),
        'undet_mean_f1_sd': np.std([x[2] for x in f1]),
        'undet_recall': np.mean([x[2] for x in rec]),
        'undet_recall_sd': np.std([x[2] for x in rec]),
        'undet_prec': np.mean([x[2] for x in prec]),
        'undet_prec_sd': np.std([x[2] for x in prec]),
        
        'overall_mean': np.mean([x[0] for x in ov_acc]),
        'overall_mean_sd': np.std([x[0] for x in ov_acc]),
        'overall_mean_f1': np.mean([x[3] for x in ov_acc]),
        'overall_mean_f1_sd': np.std([x[3] for x in ov_acc]),
        'overall_recall': np.mean([x[1] for x in ov_acc]),
        'overall_recall_sd': np.std([x[1] for x in ov_acc]),
        'overall_prec': np.mean([x[2] for x in ov_acc]),
        'overall_prec_sd': np.std([x[2] for x in ov_acc]),
        
        'runs_f1_0': np.mean([x[0] for x in splits]),
        'runs_f1_1': np.mean([x[1] for x in splits]),
        'runs_f1_2': np.mean([x[2] for x in splits]),
        
        'runs_sd_0': np.std([x[0] for x in splits]),
        'runs_sd_1': np.std([x[1] for x in splits]),
        'runs_sd_2': np.std([x[2] for x in splits])
    }
)  

import json
with open('results_fakenews_LSTM_glove_10x' + '.txt', 'w') as outfile:
  json.dump(lstm_glove_fakenews_stats, outfile)


