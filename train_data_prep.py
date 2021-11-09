import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder, scale
import numpy as np
import pandas as pd
from collections import deque
import random
import time
import os

SEQ_LEN = 25
VAL_PCT = 0.2
cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
EPOCHS = 300
BATCH_SIZE = 64
NAME = '{}-SEQ-{}'.format(SEQ_LEN, int(time.time()))
onehot_encoder = OneHotEncoder(sparse=False)


def square_features(col1, col2, col3):
    return np.sqrt(col1**2 + col2**2 + col3**2)


def preprocess_df_normalize(df):
    for col in df.columns:
        if col != 'target':
            df[col] = pd.to_numeric(df[col])
            mean = df[col].mean()
            std = df[col].std()
            # print(col, mean, std)
            df[col] = (df[col]-mean)/std
            df.dropna(inplace=True)
    df.dropna(inplace=True)
    df['squared_a'] = square_features(df['ax'].values, df['ay'].values, df['az'].values)
    df['squared_g'] = square_features(df['gx'].values, df['gy'].values, df['gz'].values)
    # print('squared_a', df['squared_a'].mean(), df['squared_a'].std())
    # print('squared_g', df['squared_g'].mean(), df['squared_g'].std())
    sequential_data = []
    prev_data = deque(maxlen=SEQ_LEN)
    for i in df.values:
        target = i[-3]
        prev_data.append([n for n in np.delete(i, -3)])  # i hasta -1, pq el ultimo dato es el target
        if len(prev_data) == SEQ_LEN:
            sequential_data.append([np.array(prev_data), target])  # y el target lo appendeo al final
    random.shuffle(sequential_data)
    return sequential_data


def split_dataset(total_data):
    X = []
    Y = []
    for seq, target in total_data:
        X.append(seq)
        Y.append(target)
    Y = np.array(Y).reshape(-1, 1)
    Y = onehot_encoder.fit_transform(Y)
    return np.array(X), Y


if __name__ == '__main__':
    path = './test_data/this_is_2/'
    main_dataset = pd.DataFrame()

    for item in os.listdir(path):
        df = pd.read_csv(path + item, names=cols, index_col=False)
        df['target'] = item[:-6]  # item es, por ej, 'circle.csv', por lo que target será 'circle'
        df.drop(0, axis=0, inplace=True)
        # df.drop('i', axis=1, inplace=True)
        main_dataset = main_dataset.append(df)
    main_dataset.reset_index()
    main_dataset = preprocess_df_normalize(main_dataset)
    random.shuffle(main_dataset)

    div_point = int(-VAL_PCT * len(main_dataset))
    train_dataset = main_dataset[:div_point]
    validation_dataset = main_dataset[div_point:]

    train_x, train_y = split_dataset(train_dataset)
    validation_x, validation_y = split_dataset(validation_dataset)

    # MODEL
'''    model = Sequential()
    model.add(LSTM(256, input_shape=(train_x.shape[1:]), activation='relu', return_sequences=True))
    model.add(Dropout(0.15))
    model.add(BatchNormalization())
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation='relu', return_sequences=True))
    model.add(Dropout(0.15))
    model.add(BatchNormalization())
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(train_y.shape[-1], activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0009, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    # tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
    checkpoint = ModelCheckpoint(f"models/m_seqlen{SEQ_LEN}_EPOCHS{EPOCHS}_BATCHS{BATCH_SIZE}.h5",
                                 monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    model.fit(np.asarray(train_x), np.asarray(train_y),
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(validation_x, validation_y),
              callbacks=[checkpoint])
    filename = 'model_addedfeatures_SEQLEN25_1.h5'
    model.save(filename)
'''