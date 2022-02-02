import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder, scale
import numpy as np
import pandas as pd
from collections import deque
import random
import time
import os
from collections import Counter

SEQ_LEN = 24
VAL_PCT = 0.2
cols = ['ax1', 'ay1', 'az1', 'gx1', 'gy1', 'gz1',
        'ax2', 'ay2', 'az2', 'gx2', 'gy2', 'gz2',
        'ax3', 'ay3', 'az3', 'gx3', 'gy3', 'gz3',
        'ax4', 'ay4', 'az4', 'gx4', 'gy4', 'gz4']
EPOCHS = 20
BATCH_SIZE = 128
categories = [['corr'], ['inco'], ['rest']]
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder.fit(categories)
columns = ['ax1', 'ay1', 'az1', 'gx1', 'gy1', 'gz1', 'sq_a1', 'sq_g1',
           'ax2', 'ay2', 'az2', 'gx2', 'gy2', 'gz2', 'sq_a2', 'sq_g2',
           'ax3', 'ay3', 'az3', 'gx3', 'gy3', 'gz3', 'sq_a3', 'sq_g3',
           'ax4', 'ay4', 'az4', 'gx4', 'gy4', 'gz4', 'sq_a4', 'sq_g4']


def square_features(col1, col2, col3):
    return np.sqrt((col1**2 + col2**2 + col3**2)/3)


def abs_variation(df: pd.DataFrame()):
    df2 = df.iloc[1:, :].reset_index(drop=True)
    for col in df.columns:
        if col != 'target':
            df[col] = df2[col] - df[col]
    return df


def preprocess_df_variation_norm(df: pd.DataFrame(), path) -> list:
    mean_std_df = pd.read_csv((path + 'mean_std.csv'), sep=',')
    axis_mean_list = list(mean_std_df.loc[0])[1:]  # saco el primer valor pq es el índice del csv, no me sirve
    axis_std_list = list(mean_std_df.loc[1])[1:]
    df.dropna(axis=0, inplace=True)
    df = df.loc[~(df == 0).all(axis=1)]  # elimina filas con todos 0s

    for col in df.columns:
        if col != 'target':
            df[col] = df[col].astype(int)

    df['sq_a1'] = square_features(df['ax1'].values, df['ay1'].values, df['az1'].values)
    df['sq_g1'] = square_features(df['gx1'].values, df['gy1'].values, df['gz1'].values)
    df['sq_a2'] = square_features(df['ax2'].values, df['ay2'].values, df['az2'].values)
    df['sq_g2'] = square_features(df['gx2'].values, df['gy2'].values, df['gz2'].values)
    df['sq_a3'] = square_features(df['ax3'].values, df['ay3'].values, df['az3'].values)
    df['sq_g3'] = square_features(df['gx3'].values, df['gy3'].values, df['gz3'].values)
    df['sq_a4'] = square_features(df['ax4'].values, df['ay4'].values, df['az4'].values)
    df['sq_g4'] = square_features(df['gx4'].values, df['gy4'].values, df['gz4'].values)

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
        target = i[-9]  # -9 es el target, pq los ultimos 8 datos son los valores al cuadrado
        prev_data.append([n for n in np.delete(i, -9)])  # deleteo el target
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
    model = Sequential()
    model.add(LSTM(256, input_shape=(train_x.shape[1:]), activation='relu', return_sequences=True, unroll=True))
    model.add(Dropout(0.15))
    model.add(BatchNormalization())
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation='relu', return_sequences=True, unroll=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation='relu', return_sequences=True, unroll=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(train_y.shape[-1], activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0007, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def gen_mean_std_file(path):
    aux_df = pd.DataFrame()
    aux_mean = []
    aux_std = []
    for item in os.listdir(path):
        dff = pd.read_csv(path + item, names=cols, index_col=False, sep=';')
        dff.dropna(inplace=True)
        dff = dff.loc[~(dff == 0).all(axis=1)]  # elimina filas con todos 0s
        dff = dff.astype(int)
        # dff = abs_variation(dff) se sacó para probar
        dff.dropna(inplace=True)
        aux_df = aux_df.append(dff)
    # aux_df.drop(0, axis=0, inplace=True)
    aux_df.reset_index(drop=True, inplace=True)
    aux_df.dropna(inplace=True)

    aux_df = aux_df.loc[~(aux_df == 0).all(axis=1)]  # elimina filas con todos 0s
    aux_df = aux_df.astype(int)
    aux_df['sq_a1'] = square_features(aux_df['ax1'].values, aux_df['ay1'].values, aux_df['az1'].values)
    aux_df['sq_g1'] = square_features(aux_df['gx1'].values, aux_df['gy1'].values, aux_df['gz1'].values)
    aux_df['sq_a2'] = square_features(aux_df['ax2'].values, aux_df['ay2'].values, aux_df['az2'].values)
    aux_df['sq_g2'] = square_features(aux_df['gx2'].values, aux_df['gy2'].values, aux_df['gz2'].values)
    aux_df['sq_a3'] = square_features(aux_df['ax3'].values, aux_df['ay3'].values, aux_df['az3'].values)
    aux_df['sq_g3'] = square_features(aux_df['gx3'].values, aux_df['gy3'].values, aux_df['gz3'].values)
    aux_df['sq_a4'] = square_features(aux_df['ax4'].values, aux_df['ay4'].values, aux_df['az4'].values)
    aux_df['sq_g4'] = square_features(aux_df['gx4'].values, aux_df['gy4'].values, aux_df['gz4'].values)

    for col in aux_df.columns:
        aux_df[col] = pd.to_numeric(aux_df[col])
        aux_mean.append(aux_df[col].mean())
        aux_std.append(aux_df[col].std())
    mean_std_df = pd.DataFrame([aux_mean, aux_std])
    mean_std_df.to_csv(path + 'mean_std.csv')
    return mean_std_df


if __name__ == '__main__':
    path = './test_data/rowing_corrector_data/'
    main_dataset = []
    mean_std = gen_mean_std_file(path)

    for item in os.listdir(path):
        if item != 'mean_std.csv':
            df = pd.read_csv(path + item, names=cols, index_col=False, sep=';')
            df['target'] = item[:4]  # item es, por ej, 'circle_12.csv', por lo que target será 'circ'
            df.drop(0, axis=0, inplace=True)
            df.reset_index()

            processed_df = preprocess_df_variation_norm(df, path)
            main_dataset = main_dataset + processed_df

    random.shuffle(main_dataset)
    print(len(main_dataset), Counter([i[-1] for i in main_dataset]))
    div_point = int(-VAL_PCT * len(main_dataset))
    train_dataset = main_dataset[:div_point]
    validation_dataset = main_dataset[div_point:]
    train_x, train_y = split_dataset(train_dataset)
    validation_x, validation_y = split_dataset(validation_dataset)

    # MODEL
    model = create_model()
    checkpoint = ModelCheckpoint("models/m12.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    start_time = time.time()
    model.fit(np.asarray(train_x), np.asarray(train_y),
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(validation_x, validation_y),
              callbacks=[checkpoint])
    filename = 'models/m12_2.h5'
    model.save(filename)
    print('Training time: {}'.format(time.time() - start_time))
