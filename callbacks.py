import pandas as pd
from dash.dependencies import Input, Output
from app import app
# from serial_read import ser
import serial
import plotly.express as px
from collections import deque
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder, scale
import random
import numpy as np
from keras.models import load_model
from train_data_prep import square_features


SEQ_LEN = 25
i = 0
model = load_model('model_addedfeatures_SEQLEN25_1.h5')
model2 = load_model('models/m_seqlen25_EPOCHS300_BATCHS64.h5')
# categories = [['res'], ['circl'], ['lin'], ['shake'], ['squar']]
categories = [['circl'], ['res'], ['squar']]
onehot_encoder = OneHotEncoder()  # sparse=False, categories=categories)  # )
onehot_encoder.fit(categories)
ser = serial.Serial(port='COM3', baudrate=115200)
X = [x for x in range(SEQ_LEN)]
count_stored_points = 0
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


def data_prep_normalize(deq):
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
    df = pd.DataFrame(deq)
    df2 = df.iloc[1:, :]
    for col in df.columns:
        df[col] = df2[col] - df[col]
        df.dropna(inplace=True)
        df[col] = scale(df[col].values)
    normalized_data = np.array(df)
    normalized_data = np.expand_dims(normalized_data, 0)
    return normalized_data  # normalized_data(transpuesto) es un array de (40,6)


@app.callback(Output('label', 'children'),
              Input('interval2', 'n_intervals'))
def get_data(values):
    global dfd, X, count_stored_points, i

    ser.reset_input_buffer()
    serialString = ser.readline()
    dato = serialString.decode('Ascii')
    dato = dato.split(sep='\t')

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
        prediction = model.predict(data_prep_normalize(dfd))
        prediction2 = model2.predict(data_prep_normalize(dfd))
        if i > 10:
            print(prediction, prediction2)
            i = 0
        i += 1
        prediction = onehot_encoder.inverse_transform(prediction)
        prediction2 = onehot_encoder.inverse_transform(prediction2)
        return '{} - {}'.format(prediction, prediction2)

    # ESTO ES PARA GUARDAR DATOS
    # file = open('./test_data/squar_2.csv', 'a')
    # try:
        # lista = list(map(int, dato))
        # file.write(''.join([str(val)+',' for val in lista])+'\n')
        # # file.close()
        # count_stored_points += 1
        # return count_stored_points


@app.callback(Output('asdasd', 'figure'),
              [Input('interval1', 'n_intervals')])
def get_data(_):
    npdfd = np.array(dfd)
    fig = make_subplots(rows=4, cols=2, subplot_titles=('Acel X', 'Gyro X', 'Acel Y', 'Gyro Y',
                                                        'Acel Z', 'Gyro Z', 'Acel^2', 'Gyro^2'))
    fig.append_trace(go.Scatter(x=X, y=npdfd[:, 0]), row=1, col=1)
    fig.append_trace(go.Scatter(x=X, y=npdfd[:, 1]), row=2, col=1)
    fig.append_trace(go.Scatter(x=X, y=npdfd[:, 2]), row=3, col=1)
    fig.append_trace(go.Scatter(x=X, y=npdfd[:, 3]), row=1, col=2)
    fig.append_trace(go.Scatter(x=X, y=npdfd[:, 4]), row=2, col=2)
    fig.append_trace(go.Scatter(x=X, y=npdfd[:, 5]), row=3, col=2)
    fig.append_trace(go.Scatter(x=X, y=npdfd[:, 6]), row=4, col=1)
    fig.append_trace(go.Scatter(x=X, y=npdfd[:, 7]), row=4, col=2)
    fig.update_layout(showlegend=False)
    return fig
