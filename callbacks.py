import os

import pandas as pd
from dash.dependencies import Input, Output, State
import dash
from app import app
# from serial_read import ser
import serial
from collections import deque
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder, scale
import random
import numpy as np
from keras.models import load_model
from train_data_prep import square_features
import time


SEQ_LEN = 12
i = 0
count_stored_points = 0
# model = load_model('models/model_addedfeatures_addedlayers_absnorm_SEQLEN15_2.h5')
model2 = load_model('models/m1639581550_removedlayersunrol_absnorm_seqlen12_EPOCHS45_BATCHS64.h5')
# categories = [['res'], ['circl'], ['lin'], ['shake'], ['squar']]
categories = [['circ'], ['rest'], ['squa']]
onehot_encoder = OneHotEncoder()  # sparse=False, categories=categories)  # )
onehot_encoder.fit(categories)

try:
    ser = serial.Serial(port='COM3', baudrate=115200)
except Exception as e:
    ser = None

X = [x for x in range(SEQ_LEN)]
columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'a_squared', 'g_squared']
axis_mean_list = [-1413.3997194950912, 3713.7786697247707, -2838.547018348624,
                  -479.115252293578, -17.25, -45.55045871559633,
                  1.3057214357366107, 0.890461854922827]  # estos datos fueron obtenidos del mean y
axis_std_list = [4860.661003906084, 5059.2175696744125, 4503.339099820826,  # la std de los datos usados para entrenar
                 3626.4416073840416, 3333.092355088295, 2681.0822506470254,
                 1.1426434991136631, 1.4855471352632454]
# dfd = pd.DataFrame(columns=columns)
dfd = deque(maxlen=SEQ_LEN)
for _ in range(SEQ_LEN):
    dfd.append([0, 0, 0, 0, 0, 0, 0, 0])

mean_std_df = pd.read_csv(('./test_data/this_is_3/' + 'mean_std.csv'))
axis_mean_list = list(mean_std_df.loc[0])[1:]  # saco el primer valor pq es el índice del csv, no me sirve
axis_std_list = list(mean_std_df.loc[1])[1:]


def data_prep_normalize(deq):
    """
    normaliza los datos como y = (y-mean)/std
    :param deq: los datos a preparar en formato deque max_len(SEQ_LEN)
    :return: datos normalizados en formato np.array
    """
    norm = lambda x: (x - mean) / std
    vf = np.vectorize(norm)
    normalized_data = np.zeros((0, SEQ_LEN))
    for i, dat in enumerate(np.transpose(deq)):  # Hago el transpose para hacer la normalización por coordenada (ax, ay...)
        mean = axis_mean_list[i]
        std = axis_std_list[i]
        normalized_data = np.append(normalized_data, [vf(dat)], axis=0)
    normalized_data = np.expand_dims(normalized_data.transpose(), 0)
    return normalized_data  # normalized_data(transpuesto) es un array de (40,6)


def data_prep_absvariation(deq):
    """
    normaliza los datos como variación absoluta con el dato anterior, escalado entre 0 y 1
    :param deq: los datos a preparar en formato deque max_len(SEQ_LEN)
    :return: datos normalizados en formato np.array
    """
    df = pd.DataFrame(deq)
    df2 = df.iloc[1:, :]
    for col in df.columns:
        df[col] = df2[col] - df[col]
        df.dropna(inplace=True)
        df[col] = scale(df[col].values)  # escala entre 0 y 1
    normalized_data = np.array(df)
    normalized_data = np.expand_dims(normalized_data, 0)
    return normalized_data


@app.callback(Output('label', 'children'),
              Input('interval2', 'n_intervals'),
              Input('checklist', 'value'))
def get_data(_, checklist_value):
    start_time = time.time()
    ctx = dash.callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]
    global dfd, X, i, count_stored_points
    if not ser:
        return 'Serial not connected'
    ser.reset_input_buffer()
    serialString = ser.readline()
    dato = serialString.decode('Ascii')
    dato = dato.split(sep='\t')

    if checklist_value == 'predict':
        # ESTO ES PARA PREDICCIONES
        try:
            lista = list(map(int, dato))
            if len(lista) == 6:
                lista.append(square_features(lista[0], lista[1], lista[2]))
                lista.append(square_features(lista[3], lista[4], lista[5]))
                dfd.append(lista)
            else:
                dfd.append(dfd[-1])
        except Exception as e:
            dfd.append(dfd[-1])

        finally:
            # prediction = model.predict(data_prep_normalize(dfd))
            prediction2 = model2.predict(data_prep_normalize(dfd))
            if i > 10:
                print(prediction2[0])  # prediction[0],
                i = 0
            i += 1
            # prediction = onehot_encoder.inverse_transform(prediction)
            prediction2 = onehot_encoder.inverse_transform(prediction2)
            print('predict time: {}'.format(time.time() - start_time))
            return '{}'.format(prediction2[0])
            # return '{}, {}'.format(prediction[0], prediction2[0])
    else:
        # ESTO ES PARA GUARDAR DATOS
        path = './test_data/this_is_3'
        n_files = [i for i in os.listdir(path) if checklist_value in i]
        # lista con todos los archivos que tienen en el nombre al tipo de señal
        n_files = [int(i.split('_')[-1].split('.')[0]) for i in n_files]
        n_files.append(0)
        n_files = max(n_files)
        if input_id == 'checklist':
            n_files += 1
            count_stored_points = 0
            print(n_files)

        filename = '{}_{}.csv'.format(checklist_value, n_files)
        file = open(path + '/' + filename, 'a')
        try:
            lista = list(map(int, dato))
            file.write(','.join([str(val) for val in lista]) + '\n')
            # file.close()
            count_stored_points += 1
            print('save time: {}'.format(time.time() - start_time))
            return count_stored_points
        except:
            return 'Cannot read'


@app.callback(Output('asdasd', 'figure'),
              [Input('interval1', 'n_intervals')])
def get_data(_):
    start_time = time.time()
    npdfd = np.array(dfd)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=X, y=npdfd[:, 6], name='Accel^2', yaxis='y'))
    fig.add_trace(go.Scattergl(x=X, y=npdfd[:, 7], name='Gyro^2', yaxis='y2'))
    fig.update_layout(showlegend=True, yaxis2={'overlaying': 'y', 'side': 'right'})
    print('plot time: {}'.format(time.time() - start_time))
    return fig
