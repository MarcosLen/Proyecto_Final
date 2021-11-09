#include <MPU9250_asukiaaa.h>

#ifdef _ESP32_HAL_I2C_H_
#define SDA_PIN 21
#define SCL_PIN 22
#endif

MPU9250_asukiaaa mySensor;
float aX, aY, aZ, aSqrt, gX, gY, gZ, mDirection, mX, mY, mZ;
String movimiento;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20000); //20seg
  while(!Serial);
  Serial.println("started");

#ifdef _ESP32_HAL_I2C_H_ // For ESP32
  Wire.begin(SDA_PIN, SCL_PIN);
  mySensor.setWire(&Wire);
#endif

  mySensor.beginAccel();
  mySensor.beginGyro();
  mySensor.beginMag();

  // You can set your own offset for mag values
  // mySensor.magXOffset = -50;
  // mySensor.magYOffset = -55;
  // mySensor.magZOffset = -10;
}

void loop() {
  uint8_t sensorId;
  if (not (mySensor.readId(&sensorId) == 0 and mySensor.accelUpdate() == 0 and mySensor.gyroUpdate() == 0)){
    Serial.println("Error al leer el sensor");
  } else {
    Serial.println("Preparese para comenzar...");
    delay(100);
    Serial.print("Ingrese el tipo de movimiento a realizar, ");
    Serial.println("pulse enter y comience a mover inmediatamente: ");
    movimiento = Serial.readStringUntil('\n');
    Serial.print("\n\n\n");
    Serial.println("accelX; accelY; accelZ; accelSqrt; gyroX; gyroY; gyroZ; mov");
    for(int i=0; i<30; i++){
      aX = mySensor.accelX();
      aY = mySensor.accelY();
      aZ = mySensor.accelZ();
      aSqrt = mySensor.accelSqrt();
      gX = mySensor.gyroX();
      gY = mySensor.gyroY();
      gZ = mySensor.gyroZ();
      Serial.println(String(aX) + ";" + String(aY) + ";" + String(aZ) + ";" +
                     String(aSqrt) + ";" + String(gX) + ";" + String(gY) +
                     ";" + String(gZ) + ";" + movimiento);
      delay(300);
      }
  }   
delay(500);
}
