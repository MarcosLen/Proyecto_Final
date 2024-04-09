import serial
import sys
from time import time
from train_data_prep import square_features, onehot_encoder
from keras.models import load_model
import numpy as np
from collections import deque


def data_prep_absvariation(dato_actual: list, dato_anterior: list) -> list:
    dat = [d_act - d_ant for d_act, d_ant in zip(dato_actual, dato_anterior)]
    return dat


model = load_model('models/modelo4.h5')
SEQ_LEN = 15
dfd = deque(maxlen=SEQ_LEN)
for _ in range(SEQ_LEN):
    dfd.append([0, 0, 0, 0, 0, 0, 0, 0])
dato_aux = [0, 0, 0, 0, 0, 0, 0, 0]

# path = './test_data/Datos_2024'
# filename = 'Tiempo_de_almacenamiento.csv'
# file = open(path + '/' + filename, 'a')

axis_mean_list = [141.97986785050588,-2634.475428453438,16864.872289902953,12.091472227957878,-5.242308486475325,57.1180053685732,10269.127921385387,1781.9547048505435]
axis_std_list = [3040.588252378687,4275.346921054399,1669.0948800590704,1697.8194996760365,1598.9919660253863,3835.7433284013296,1332.4961872902713,1840.5732652143784]


ser = serial.Serial(port='COM6', baudrate=115200, timeout=1)
ser.reset_input_buffer()
for _ in range(50):
    ser.write(b'\n')
    a = ser.readline()
    print(sys.getsizeof(a))


# LECTURA DE UN DATO
# while True:
#     # start = time()
#     ser.write(b'\n')
#     serialString = ser.readline()
#     dato = serialString.decode('Ascii')
#     lista = dato.split(sep='\t')
#     lista = list(map(int, lista))
#     print(lista)
#     # print(time()-start)

start = time()
for i in range(1000):
    ser.write(b'\n')
    a = ser.readline()
# TIEMPO IMPLICADO EN LA LECTURA DE UN DATO (correr programa como está)


# TIEMPO IMPLICADO EN EL ALMACENAMIENTO DE UN DATO
#     dato = a.decode('Ascii')
#     dato = dato.split(sep='\t')
#     lista = list(map(int, dato))
#     file.write(';'.join([str(val) for val in lista]) + '\n')


# TIEMPO IMPLICADO EN LA CLASIFICACIÓN DE LA RED
    dato = a.decode('Ascii')
    dato = dato.split(sep='\t')
    lista = list(map(int, dato))

    lista.append(square_features(lista[0], lista[1], lista[2]))
    lista.append(square_features(lista[3], lista[4], lista[5]))

    lista = [(x - axis_mean_list[n]) / axis_std_list[n] for n, x in enumerate(lista)]
    lista_norm = data_prep_absvariation(lista, dato_aux)
    dato_aux = lista
    dfd.append(lista_norm)

    prediction = model.predict(np.expand_dims(np.array(dfd), 0))

    prediction = onehot_encoder.inverse_transform(prediction)

print("tiempo: ", (time() - start)/1000)
