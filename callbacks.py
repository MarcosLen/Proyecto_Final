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
from train_data_prep import square_features, SEQ_LEN
import time


i = 0
count_stored_points = 0
# model = load_model('models/m9.h5')
model = load_model('models/m10.h5')
# categories = [['circ'], ['rest'], ['squa']]
categories = [['Descanso'], ['Técnica Correcta'], ['Técnica Incorrecta']]
onehot_encoder = OneHotEncoder()  # sparse=False, categories=categories)  # )
onehot_encoder.fit(categories)

try:
    ser = serial.Serial(port='COM3', baudrate=74880)  # 115200, 38400
except Exception as e:
    ser = None

X = [x for x in range(SEQ_LEN)]
dfd = deque(maxlen=SEQ_LEN)
for _ in range(SEQ_LEN):
    dfd.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
dato_aux = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 32 0s

mean_std_df = pd.read_csv(('./test_data/rowing_corrector_data/' + 'mean_std.csv'))  # leo los mean y std de todas las mediciones del
                                                                        # archivo que generó al entrenar el modelo
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


def data_prep_absvariation(dato_actual: list, dato_anterior: list) -> list:
    """
    normaliza los datos como variación absoluta con respecto al dato anterior
    :param dato_actual: lista con los datos leídos en el interrupt actual
    :param dato_anterior: lista con los datos leídos en el interrupt anterior
    :return: dato actual en el formato correcto, enviado como lista
    """
    dato = [d_act - d_ant for d_act, d_ant in zip(dato_actual, dato_anterior)]
    return dato


@app.callback(Output('label', 'children'),
              Input('interval2', 'n_intervals'),
              Input('checklist', 'value'))
def get_data(_, checklist_value):
    ctx = dash.callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]
    global dfd, X, i, count_stored_points, dato_aux
    if not ser:
        return 'Serial not connected'
    ser.reset_input_buffer()
    serialString = ser.readline()
    serialString = ser.readline()
    dato = serialString.decode('Ascii')
    dato = dato.split(sep='\t')
    if checklist_value == 'predict':
        # ESTO ES PARA PREDICCIONES
        try:
            lista = list(map(int, dato))
            if len(lista) == 24:
                lista.append(square_features(lista[0], lista[1], lista[2]))  # sq_a1
                lista.append(square_features(lista[3], lista[4], lista[5]))  # sq_g1
                lista.append(square_features(lista[6], lista[7], lista[8]))  # sq_a2
                lista.append(square_features(lista[9], lista[10], lista[11]))  # sq_g2
                lista.append(square_features(lista[12], lista[13], lista[14]))  # sq_a3
                lista.append(square_features(lista[15], lista[16], lista[17]))  # sq_g3
                lista.append(square_features(lista[18], lista[19], lista[20]))  # sq_a4
                lista.append(square_features(lista[21], lista[22], lista[23]))  # sq_g4
                lista = [(x-axis_mean_list[n])/axis_std_list[n] for n, x in enumerate(lista)]
                lista_norm = data_prep_absvariation(lista, dato_aux)
                dato_aux = lista
                dfd.append(lista_norm)
            else:
                dfd.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        except Exception as e:
            dfd.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        finally:
            prediction = model.predict(np.expand_dims(np.array(dfd), 0))

            # if i > 10:
            #     print(prediction[0], lista, lista_norm)
            #     i = 0
            # i += 1

            prediction = onehot_encoder.inverse_transform(prediction)
            return 'Predicción: {}'.format(prediction[0])
    else:
        # ESTO ES PARA GUARDAR DATOS
        path = './test_data/rowing_corrector_data'
        n_files = [i for i in os.listdir(path) if checklist_value in i]
        # lista con todos los archivos que tienen en el nombre al tipo de señal
        n_files = [int(i.split('_')[-1].split('.')[0]) for i in n_files]
        n_files.append(0)
        n_files = max(n_files)

        if input_id == 'checklist':
            n_files += 1
            count_stored_points = 0

        filename = '{}_{}.csv'.format(checklist_value, n_files)
        file = open(path + '/' + filename, 'a')
        try:
            lista = list(map(int, dato))
            file.write(';'.join([str(val) for val in lista]) + '\n')
            # file.close()
            count_stored_points += 1
            return count_stored_points
        except:
            return 'Cannot read'


@app.callback(Output('asdasd', 'figure'),
              [Input('interval1', 'n_intervals')])
def get_data(_):
    npdfd = np.array(dfd)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=X, y=npdfd[:, 24], name='S1', yaxis='y'))
    fig.add_trace(go.Scattergl(x=X, y=npdfd[:, 26], name='S2', yaxis='y'))
    fig.add_trace(go.Scattergl(x=X, y=npdfd[:, 28], name='S3', yaxis='y'))
    fig.add_trace(go.Scattergl(x=X, y=npdfd[:, 30], name='S4', yaxis='y'))
    fig.update_layout(showlegend=True)
    return fig
