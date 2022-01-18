import serial

ser = serial.Serial(port='COM3', baudrate=115200)

while True:
    # Read data out of the buffer until a carraige return / new line is found
    serialString = ser.readline()

    # Print the contents of the serial data
    dato = serialString.decode('Ascii')
    print(dato)
    lista = dato.split(sep='\t')
    lista = list(map(int, lista))
    print(lista)
