import serial
from time import time

ser = serial.Serial(port='COM3', baudrate=74880)  # 115200 , 38400

while True:
    start = time()
    # Read data out of the buffer until a carraige return / new line is found
    # ser.reset_input_buffer()
    serialString = ser.readline()
    # serialString = ser.readline()

    dato = serialString.decode('Ascii')
    lista = dato.split(sep='\t')
    lista = list(map(int, lista))
    print(time()-start, len(lista), lista)
