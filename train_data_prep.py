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
from collections import Counter

SEQ_LEN = 20
VAL_PCT = 0.2
cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
EPOCHS = 45
BATCH_SIZE = 64
NAME = '{}-SEQ-{}-TIME'.format(SEQ_LEN, int(time.time()))
categories = [['circ'], ['rest'], ['squa']]
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder.fit(categories)
columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'squared_a', 'squared_g']


def square_features(col1, col2, col3):
    return np.sqrt(col1**2 + col2**2 + col3**2)


def preprocess_df_normalize(df, path) -> list:
    mean_std_df = pd.read_csv((path + 'mean_std.csv'))
    axis_mean_list = list(mean_std_df.loc[0])[1:]  # saco el primer valor pq es el índice del csv, no me sirve
    axis_std_list = list(mean_std_df.loc[1])[1:]

    df['squared_a'] = square_features(df['ax'].values, df['ay'].values, df['az'].values)
    df['squared_g'] = square_features(df['gx'].values, df['gy'].values, df['gz'].values)

    for n, col in enumerate(df[columns]):
        df[col] = pd.to_numeric(df[col])
        df[col] = (df[col]-axis_mean_list[n])/axis_std_list[n]
        df.dropna(inplace=True)
    df.dropna(inplace=True)

    sequential_data = []
    prev_data = deque(maxlen=SEQ_LEN)
    for i in df.values:
        target = i[-3]  # el target está en la posición -3 de la fila del df
        prev_data.append([n for n in np.delete(i, -3)])  # deleteo el target
        if len(prev_data) == SEQ_LEN:
            sequential_data.append([np.array(prev_data), target])  # y el target lo appendeo al final
    random.shuffle(sequential_data)
    return sequential_data


def abs_variation(df: pd.DataFrame()):
    df2 = df.iloc[1:, :].reset_index(drop=True)
    for col in df.columns:
        if col != 'target':
            df[col] = df2[col] - df[col]
    return df


def preprocess_df_variation_norm(df: pd.DataFrame(), path) -> list:
    mean_std_df = pd.read_csv((path + 'mean_std.csv'))
    axis_mean_list = list(mean_std_df.loc[0])[1:]  # saco el primer valor pq es el índice del csv, no me sirve
    axis_std_list = list(mean_std_df.loc[1])[1:]
    df['squared_a'] = square_features(df['ax'].values, df['ay'].values, df['az'].values)
    df['squared_g'] = square_features(df['gx'].values, df['gy'].values, df['gz'].values)

    for n, col in enumerate(df[columns]):
        df[col] = pd.to_numeric(df[col])
        df[col] = (df[col] - axis_mean_list[n]) / axis_std_list[n]
        df.dropna(inplace=True)
    df.dropna(inplace=True)

    df = abs_variation(df)
    df.dropna(inplace=True)

    sequential_data = []
    prev_data = deque(maxlen=SEQ_LEN)
    for i in df.values:
        target = i[-3]
        prev_data.append([n for n in np.delete(i, -3)])  # deleteo el target
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
    Y = onehot_encoder.transform(Y)
    return np.array(X), Y


def create_model():
    model = Sequential()  # abajo, unroll es para aumentar la velocidad de la prediccion
    model.add(LSTM(256, input_shape=(train_x.shape[1:]), activation='relu', return_sequences=True, unroll=True))
    model.add(Dropout(0.15))
    model.add(BatchNormalization())
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation='relu', return_sequences=True, unroll=True))
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
    return model


def gen_mean_std_file(path):
    aux_df = pd.DataFrame()
    aux_mean = []
    aux_std = []
    for item in os.listdir(path):
        dff = pd.read_csv(path + item, names=cols, index_col=False)
        aux_df = aux_df.append(dff)
    # aux_df.drop(0, axis=0, inplace=True)
    aux_df.reset_index(drop=True, inplace=True)
    aux_df.dropna(inplace=True)
    aux_df['squared_a'] = square_features(aux_df['ax'].values, aux_df['ay'].values, aux_df['az'].values)
    aux_df['squared_g'] = square_features(aux_df['gx'].values, aux_df['gy'].values, aux_df['gz'].values)
    for col in aux_df.columns:
        aux_df[col] = pd.to_numeric(aux_df[col])
        aux_mean.append(aux_df[col].mean())
        aux_std.append(aux_df[col].std())
    mean_std_df = pd.DataFrame([aux_mean, aux_std])
    mean_std_df.to_csv(path + 'mean_std.csv')
    return mean_std_df


if __name__ == '__main__':
    path = './test_data/this_is_4/'
    main_dataset = []
    mean_std = gen_mean_std_file(path)

    for item in os.listdir(path):
        df = pd.read_csv(path + item, names=cols, index_col=False)
        df['target'] = item[:4]  # item es, por ej, 'circle_12.csv', por lo que target será 'circ'
        df.drop(0, axis=0, inplace=True)
        # df.drop('i', axis=1, inplace=True)  # esta linea es para el dataset1 que tenía indice en el csv
        df.reset_index()
        processed_df = preprocess_df_variation_norm(df, path)
        main_dataset = main_dataset + processed_df

    # print(len(main_dataset), Counter([i[-1] for i in main_dataset]))
    random.shuffle(main_dataset)
    div_point = int(-VAL_PCT * len(main_dataset))
    train_dataset = main_dataset[:div_point]
    validation_dataset = main_dataset[div_point:]
    train_x, train_y = split_dataset(train_dataset)
    validation_x, validation_y = split_dataset(validation_dataset)

    # MODEL
    # model = create_model()
    # # tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
    # checkpoint = ModelCheckpoint(f"models/m6.h5",
    #                              monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # start_time = time.time()
    # model.fit(np.asarray(train_x), np.asarray(train_y),
    #           batch_size=BATCH_SIZE,
    #           epochs=EPOCHS,
    #           validation_data=(validation_x, validation_y),
    #           callbacks=[checkpoint])
    # filename = 'models/m6_2.h5'.format(int(time.time()), SEQ_LEN)
    # model.save(filename)
    # print('Training time: {}'.format(time.time() - start_time))
