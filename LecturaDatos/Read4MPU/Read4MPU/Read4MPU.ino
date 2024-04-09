// SDA --> A4
// SCL --> A5

#include "I2Cdev.h"
#include "MPU6050.h"
#include "Wire.h"

MPU6050 mpu;

int16_t ax, ay, az;
int16_t gx, gy, gz;

#define ADO1_PIN 5
#define ADO2_PIN 6
#define ADO3_PIN 2
#define ADO4_PIN 8
// AD0 low = 0x68 (default)
// AD0 high = 0x69

void iniciarSensores(){
    digitalWrite(ADO1_PIN, LOW);
    digitalWrite(ADO2_PIN, HIGH);
    digitalWrite(ADO3_PIN, HIGH);
    digitalWrite(ADO4_PIN, HIGH);
    mpu.initialize();
    
    digitalWrite(ADO1_PIN, HIGH);
    digitalWrite(ADO2_PIN, LOW);
    mpu.initialize();
    
    digitalWrite(ADO2_PIN, HIGH);
    digitalWrite(ADO3_PIN, LOW);
    mpu.initialize();
    
    digitalWrite(ADO3_PIN, HIGH);
    digitalWrite(ADO4_PIN, LOW);
    mpu.initialize();
  }


void setup() {
    pinMode(ADO1_PIN, OUTPUT);
    pinMode(ADO2_PIN, OUTPUT);
    pinMode(ADO3_PIN, OUTPUT);
    pinMode(ADO4_PIN, OUTPUT);
    mpu.initialize();
    iniciarSensores();
    
    Serial.begin(115200);
    Wire.setClock(400000);
    Wire.begin();
    Wire.setClock(400000);
    mpu.initialize();
}

void loop() {
    digitalWrite(ADO4_PIN, HIGH);
    digitalWrite(ADO1_PIN, LOW);
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    Serial.print(ax); Serial.print("\t");
    Serial.print(ay); Serial.print("\t");
    Serial.print(az); Serial.print("\t");
    Serial.print(gx); Serial.print("\t");
    Serial.print(gy); Serial.print("\t");
    Serial.print(gz); Serial.print("\t");


    digitalWrite(ADO1_PIN, HIGH);
    digitalWrite(ADO2_PIN, LOW);
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    Serial.print(ax); Serial.print("\t");
    Serial.print(ay); Serial.print("\t");
    Serial.print(az); Serial.print("\t");
    Serial.print(gx); Serial.print("\t");
    Serial.print(gy); Serial.print("\t");
    Serial.print(gz); Serial.print("\t");

    
    digitalWrite(ADO2_PIN, HIGH);
    digitalWrite(ADO3_PIN, LOW);
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    Serial.print(ax); Serial.print("\t");
    Serial.print(ay); Serial.print("\t");
    Serial.print(az); Serial.print("\t");
    Serial.print(gx); Serial.print("\t");
    Serial.print(gy); Serial.print("\t");
    Serial.print(gz); Serial.print("\t");


    digitalWrite(ADO3_PIN, HIGH);
    digitalWrite(ADO4_PIN, LOW);
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    Serial.print(ax); Serial.print("\t");
    Serial.print(ay); Serial.print("\t");
    Serial.print(az); Serial.print("\t");
    Serial.print(gx); Serial.print("\t");
    Serial.print(gy); Serial.print("\t");
    Serial.println(gz);
}
