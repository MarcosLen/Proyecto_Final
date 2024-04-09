import os
import pandas as pd
from dash.dependencies import Input, Output
import dash
from app import app
import serial
from collections import deque
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.models import load_model
from train_data_prep import square_features, SEQ_LEN
import dash_html_components as html


i = 0
count_stored_points = 0
model = load_model('models/modelo4.h5')
categories = [['circular'], ['cuadrado'], ['reposo']]
cat_dict = {'circular': 'MOVIMIENTO:\t\tCIRCULAR',
            'cuadrado': 'MOVIMIENTO:\t\tCUADRADO',
            'reposo': 'MOVIMIENTO:\t\tREPOSO'}
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(categories)

try:
    ser = serial.Serial(port='COM6', baudrate=115200, timeout=1)
    ser.reset_input_buffer()
    for _ in range(10):
        ser.write(b'\n')
        a = ser.readline()
    ser.reset_input_buffer()
except Exception as e:
    ser = None

X = [x for x in range(SEQ_LEN)]
dfd = deque(maxlen=SEQ_LEN)
for _ in range(SEQ_LEN):
    dfd.append([0, 0, 0, 0, 0, 0, 0, 0])
dato_aux = [0, 0, 0, 0, 0, 0, 0, 0]  # 8 0s


prom_prediction_deq = deque(maxlen=10)
for _ in range(10):
    prom_prediction_deq.append(np.array([0, 0, 0]).reshape(1, 3))


mean_std_df = pd.read_csv(('./test_data/Datos_2024/' + 'mean_std.csv'))  # leo los mean y std de todas las mediciones
                                                                        # del archivo que generó al entrenar el modelo
axis_mean_list = list(mean_std_df.loc[0])[1:]  # saco el primer valor pq es el índice del csv, no me sirve
axis_std_list = list(mean_std_df.loc[1])[1:]


def data_prep_absvariation(dato_actual: list, dato_anterior: list) -> list:
    dato = [d_act - d_ant for d_act, d_ant in zip(dato_actual, dato_anterior)]
    return dato


def prom_prediction(deq):
    deq_array = np.array(deq)
    prediccion_promedio = [i.mean() for i in deq_array.transpose()]
    return np.array(prediccion_promedio).reshape(1, 3)


@app.callback(Output('texto-principal', 'children'),
              Input('interval2', 'n_intervals'),
              Input('checklist', 'value'),
              Input('modo_admin', 'value'))
def get_data(_, checklist_value, val):
    ctx = dash.callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]
    global dfd, X, i, count_stored_points, dato_aux
    if not ser:
        return [html.H1("Serial not connected", style={'font-size': 47})]

    ser.write(b'\n')
    serialString = ser.readline()
    dato = serialString.decode('Ascii')
    dato = dato.split(sep='\t')
    if 'Modo Admin' in val:
        if checklist_value == 'nada':
            return [html.H1("Seleccione qué datos va a ingresar", style={'font-size': 47})]
        else:
            # ESTO ES PARA GUARDAR DATOS
            path = './test_data/Datos_2024'
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
                texto = "Ingresando datos de movimiento {}. Cantidad: {}".format(checklist_value, count_stored_points)
                return [html.H1(texto, style={'font-size': 47})]
            except:
                return [html.H1("NO SE PUEDE ACCEDER AL PUERTO SERIE", style={'font-size': 47})]
    else:
        # ESTO ES PARA PREDICCIONES
        try:
            incidente = list(map(int, dato))
            if len(incidente) == 6:

                incidente.append(square_features(incidente[0], incidente[1], incidente[2]))  # sq_a1
                incidente.append(square_features(incidente[3], incidente[4], incidente[5]))  # sq_g1
                incidente = [(x - axis_mean_list[n]) / axis_std_list[n] for n, x in enumerate(incidente)]

                incidente_norm = data_prep_absvariation(incidente, dato_aux)
                dato_aux = incidente
                dfd.append(incidente_norm)
            else:
                dfd.append([0, 0, 0, 0, 0, 0, 0, 0])
        except Exception as e:
            dfd.append([0, 0, 0, 0, 0, 0, 0, 0])

        finally:
            prediction = model.predict(np.expand_dims(np.array(dfd), 0))
            prom_prediction_deq.append(prediction)
            prediction = prom_prediction(prediction)

            texto1 = 'Circular: {:.2f}%'.format(prediction[0][0] * 100)
            texto2 = 'Cuadrado: {:.2f}%'.format(prediction[0][1] * 100)
            texto3 = 'Reposo:   {:.2f}%'.format(prediction[0][2] * 100)
            estilo1 = {'font-size': 47, 'color': 'BLUE'}
            estilo2 = {'font-size': 27, 'color': 'BLACK'}
            return [html.H1(texto1, style=estilo1 if prediction[0][0] == max(prediction[0]) else estilo2),
                    html.H1(texto2, style=estilo1 if prediction[0][1] == max(prediction[0]) else estilo2),
                    html.H1(texto3, style=estilo1 if prediction[0][2] == max(prediction[0]) else estilo2)]


@app.callback(Output('checklist-div', 'style'),
              Output('titulo', 'style'),
              Output('titulo', 'children'),
              Output('checklist', 'value'),
              Input('modo_admin', 'value'))
def get_data(val):
    mostrar = {'display': 'block'} if 'Modo Admin' in val else {'display': 'none'}
    estilo = {'color': 'BLUE'} if 'Modo Admin' in val else {'color': '#5a5a5a'}
    titulo = 'MODO ADMINISTRADOR' if 'Modo Admin' in val else 'Proyecto final'
    return mostrar, estilo, titulo, 'nada'
