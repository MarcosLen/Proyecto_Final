//GND - GND
//VCC - VCC
//SDA - Pin A4
//SCL - Pin A5

#include <Wire.h>

#define    MPU9250_ADDRESS            0x68
#define    MAG_ADDRESS                0x0C

#define    GYRO_FULL_SCALE_250_DPS    0x00  
#define    GYRO_FULL_SCALE_500_DPS    0x08
#define    GYRO_FULL_SCALE_1000_DPS   0x10
#define    GYRO_FULL_SCALE_2000_DPS   0x18

#define    ACC_FULL_SCALE_2_G        0x00  
#define    ACC_FULL_SCALE_4_G        0x08
#define    ACC_FULL_SCALE_8_G        0x10
#define    ACC_FULL_SCALE_16_G       0x18

String movimiento;


//Funcion auxiliar lectura
void I2Cread(uint8_t Address, uint8_t Register, uint8_t Nbytes, uint8_t* Data)
{
   Wire.beginTransmission(Address);
   Wire.write(Register);
   Wire.endTransmission();

   Wire.requestFrom(Address, Nbytes);
   uint8_t index = 0;
   while (Wire.available())
      Data[index++] = Wire.read();
}


// Funcion auxiliar de escritura
void I2CwriteByte(uint8_t Address, uint8_t Register, uint8_t Data)
{
   Wire.beginTransmission(Address);
   Wire.write(Register);
   Wire.write(Data);
   Wire.endTransmission();
}


void setup()
{
   Wire.begin();
   Serial.begin(115200);
   Serial.setTimeout(20000); //20seg
   // while(!Serial);
   // Serial.println("started");
   // Configurar acelerometro
   I2CwriteByte(MPU9250_ADDRESS, 28, ACC_FULL_SCALE_16_G);
   // Configurar giroscopio
   I2CwriteByte(MPU9250_ADDRESS, 27, GYRO_FULL_SCALE_2000_DPS);
   // Configurar magnetometro
   I2CwriteByte(MPU9250_ADDRESS, 0x37, 0x02);
   I2CwriteByte(MAG_ADDRESS, 0x0A, 0x01);
}


void loop()
{
  //Serial.println("Preparese para comenzar...");
  //delay(500);
  //Serial.print("Ingrese el tipo de movimiento a realizar, ");
  //Serial.println("pulse enter y comience a mover inmediatamente: ");
  //delay(100);
  //movimiento = Serial.readStringUntil('\n');
  //Serial.print("\n\n\n");
  //Serial.println("accelX\taccelY\taccelZ\tgyroX\tgyroY\tgyroZ\tmov");
  for(int i=0; i<30; i++){
     // ---  Lectura acelerometro y giroscopio --- 
     uint8_t Buf[14];
     I2Cread(MPU9250_ADDRESS, 0x3B, 14, Buf);
  
     // Convertir registros acelerometro
     int16_t ax = -(Buf[0] << 8 | Buf[1]);
     int16_t ay = -(Buf[2] << 8 | Buf[3]);
     int16_t az = Buf[4] << 8 | Buf[5];
    
     // Convertir registros giroscopio
     int16_t gx = -(Buf[8] << 8 | Buf[9]);
     int16_t gy = -(Buf[10] << 8 | Buf[11]);
     int16_t gz = Buf[12] << 8 | Buf[13];
    
      
     // ---  Lectura del magnetometro --- 
     uint8_t ST1;
     do
     {
        I2Cread(MAG_ADDRESS, 0x02, 1, &ST1);
     } while (!(ST1 & 0x01));
          uint8_t Mag[7];
     I2Cread(MAG_ADDRESS, 0x03, 7, Mag);
       
     // --- Mostrar valores ---
          // Acelerometro
     Serial.print(ax, DEC);
     Serial.print("\t");
     Serial.print(ay, DEC);
     Serial.print("\t");
     Serial.print(az, DEC);
     Serial.print("\t");
          // Giroscopio
     Serial.print(gx, DEC);
     Serial.print("\t");
     Serial.print(gy, DEC);
     Serial.print("\t");
     Serial.print(gz, DEC);
     Serial.println();
    
     //Serial.println(movimiento);
     delay(100);    
    }
   //delay(500);
}
