// SDA --> A4
// SCL --> A5

#include "MPU6050.h"
#include "Wire.h"


MPU6050 mpu;
int16_t ax, ay, az;
int16_t gx, gy, gz;
bool stringComplete = false;


void setup() {
  Serial.begin(115200);
  Wire.setClock(400000);
  Wire.begin();
  Wire.setClock(400000);
  mpu.initialize();
  mpu.CalibrateGyro(7);
  mpu.CalibrateAccel(7);
}


void loop() {
  if (stringComplete) {
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    Serial.print(ax); Serial.print("\t");
    Serial.print(ay); Serial.print("\t");
    Serial.print(az); Serial.print("\t");
    Serial.print(gx); Serial.print("\t");
    Serial.print(gy); Serial.print("\t");
    Serial.print(gz); Serial.print("\n");
    stringComplete = false;
  }
  
}


void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar=='\n'){
      stringComplete = true;
    }
  }
}