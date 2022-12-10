import time

import keras
import numpy as np
import pandas as pd
from pathlib import Path
import json

import sklearn.metrics
import tensorflow as tf
from halo import Halo
from keras.layers import Dense
from keras import Model
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

from sklearn import svm
from sklearn.naive_bayes import CategoricalNB
import pickle

import numerapi
import gc
import utils
from utils import (
    save_model,
    load_model,
    get_biggest_change_features,
    validation_metrics,
    ERA_COL,
    DATA_TYPE_COL,
    TARGET_COL
)
start = time.time()
napi = numerapi.NumerAPI()

utils.download_data()  # Training and validation
# Also get live data
current_round = napi.get_current_round(tournament=8)
Path("./v4").mkdir(parents=False, exist_ok=True)
napi.download_dataset("v4/live_int8.parquet", dest_path=f"./data/live_{current_round}_int8.parquet")

print('Reading minimal training data')
# read the feature metadata and get a feature set (or all the features)
with open("./data/features.json", "r") as f:
    feature_metadata = json.load(f)
features = feature_metadata["feature_sets"]["medium"] # get the medium feature set
# read in just those features along with era and target columns
read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]

# note: sometimes when trying to read the downloaded data you get an error about invalid magic parquet bytes...
# if so, delete the file and rerun the napi.download_dataset to fix the corrupted file
training_data = pd.read_parquet('./data/train_int8.parquet',
                                columns=read_columns)
validation_data = pd.read_parquet('./data/validation_int8.parquet',
                                  columns=read_columns)
live_data = pd.read_parquet(f'./data/live_{current_round}_int8.parquet',
                                  columns=read_columns)
print('Finished reading data')

# pare down the number of eras to every 4th era
every_4th_era = training_data[ERA_COL].unique()[::4]
training_data = training_data[training_data[ERA_COL].isin(every_4th_era)]
# training_data["erano"] = training_data.era.astype(int)
X_train = training_data.filter(like='feature_', axis='columns')
Y_train = training_data[TARGET_COL]
X_val = validation_data.filter(like='feature_', axis='columns')
# Y_val = validation_data[TARGET_COL]

all_feature_corrs = training_data.groupby(ERA_COL).apply(
    lambda era: era[features].corrwith(era[TARGET_COL])
)
riskiest_features = get_biggest_change_features(all_feature_corrs, 50)
gc.collect()

# Building the encoder
data_dim = len(features)
encoding_dim = 40
input_data = keras.Input(shape=(data_dim, ))
# Encoder layers
e1 = Dense(encoding_dim * 5, activation='relu')(input_data)
e1 = keras.layers.BatchNormalization()(e1)
e2 = Dense(encoding_dim, activation='sigmoid')(e1)
e2 = keras.layers.BatchNormalization()(e2)

# bottleneck
bottleneck = Dense(encoding_dim)(e2)

# Decoder, not used rn
encoded_input = keras.Input(shape=(encoding_dim,))
d1 = Dense(encoding_dim, activation='relu')(bottleneck)
d1 = keras.layers.BatchNormalization()(d1)
d2 = Dense(encoding_dim * 5, activation='sigmoid')(d1)

decoder = keras.Model(encoded_input, encoded_input)
output = Dense(data_dim)(d2)

encoder = Model(input_data, bottleneck)
encoder.compile(optimizer='adam', loss='mse')

es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=1500,
                                      restore_best_weights=True)
X_train_tf = tf.convert_to_tensor(X_train)
X_val_tf = tf.convert_to_tensor(X_val)

"""encoder.fit(X_train_tf, X_train_tf, epochs=5, batch_size=X_train.shape[0] // 100,
                shuffle=True, callbacks=es, validation_data=(X_val_tf, X_val_tf))"""

# SVM
spinner = Halo(text='', spinner='dots')
spinner.start("starting svm\n")
# preprocessing
# subset X_train
X_train, X_rest, Y_train, Y_rest = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=0.9)
X_val, X_rest, Y_val, Y_rest = sklearn.model_selection.train_test_split(X_rest, Y_rest, test_size=0.9)

# Standardize train data
t = StandardScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_val = t.transform(X_val)

# Feed to encoder to reduce dimension
X_train = encoder.predict(X_train)
X_val = encoder.predict(X_val)
print(X_train.shape)

# label encoder
le = LabelEncoder()
le.fit(Y_train)
Y_train = le.transform(Y_train)
Y_val = le.transform(Y_val)

# Training model
model = svm.SVC()
model.fit(X_train, Y_train)
Y_hat = model.predict(X_val)
spinner.succeed()
score = accuracy_score(Y_val, Y_hat)
print("score:", score)


print(f'Time elapsed: {(time.time() - start) / 60} mins')
gc.collect()
