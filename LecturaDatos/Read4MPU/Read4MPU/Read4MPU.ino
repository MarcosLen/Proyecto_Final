// SDA --> A4
// SCL --> A5

#include "I2Cdev.h"
#include "MPU6050.h"

// Arduino Wire library is required if I2Cdev I2CDEV_ARDUINO_WIRE implementation
// is used in I2Cdev.h
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    #include "Wire.h"
#endif

// class default I2C address is 0x68
// specific I2C addresses may be passed as a parameter here
// AD0 low = 0x68 (default for InvenSense evaluation board)
// AD0 high = 0x69
MPU6050 accelgyro;

int16_t ax, ay, az;
int16_t gx, gy, gz;

// uncomment "OUTPUT_READABLE_ACCELGYRO" if you want to see a tab-separated
// list of the accel X/Y/Z and then gyro X/Y/Z values in decimal. Easy to read,
// not so easy to parse, and slow(er) over UART.
#define OUTPUT_READABLE_ACCELGYRO

// uncomment "OUTPUT_BINARY_ACCELGYRO" to send all 6 axes of data as 16-bit
// binary, one right after the other. This is very fast (as fast as possible
// without compression or data loss), and easy to parse, but impossible to read
// for a human.
// #define OUTPUT_BINARY_ACCELGYRO


#define ADO1_PIN 5
#define ADO2_PIN 6
#define ADO3_PIN 2
#define ADO4_PIN 8


void iniciarSensores(){
    digitalWrite(ADO1_PIN, LOW);
    digitalWrite(ADO2_PIN, HIGH);
    digitalWrite(ADO3_PIN, HIGH);
    digitalWrite(ADO4_PIN, HIGH);
    accelgyro.setSleepEnabled(false);
    
    digitalWrite(ADO1_PIN, HIGH);
    digitalWrite(ADO2_PIN, LOW);
    accelgyro.setSleepEnabled(false);

    digitalWrite(ADO2_PIN, HIGH);
    digitalWrite(ADO3_PIN, LOW);
    accelgyro.setSleepEnabled(false);
    
    digitalWrite(ADO3_PIN, HIGH);
    digitalWrite(ADO4_PIN, LOW);
    accelgyro.setSleepEnabled(false);
    
    digitalWrite(ADO4_PIN, HIGH);
    digitalWrite(ADO1_PIN, LOW);
  }


void setup() {
    pinMode(ADO1_PIN, OUTPUT);
    pinMode(ADO2_PIN, OUTPUT);
    pinMode(ADO3_PIN, OUTPUT);
    pinMode(ADO4_PIN, OUTPUT);

    accelgyro.initialize();
    
    Serial.begin(74880);
    // join I2C bus (I2Cdev library doesn't do this automatically)
    #if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
        Wire.begin();
    #elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
        Fastwire::setup(400, true);
    #endif

    iniciarSensores();
}

void loop() {
    digitalWrite(ADO4_PIN, HIGH);
    digitalWrite(ADO1_PIN, LOW);
    // read raw accel/gyro measurements from device
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    delay(3);
    #ifdef OUTPUT_READABLE_ACCELGYRO
        Serial.print(ax); Serial.print("\t");
        Serial.print(ay); Serial.print("\t");
        Serial.print(az); Serial.print("\t");
        Serial.print(gx); Serial.print("\t");
        Serial.print(gy); Serial.print("\t");
        Serial.print(gz); Serial.print("\t");
    #endif

    digitalWrite(ADO1_PIN, HIGH);
    digitalWrite(ADO2_PIN, LOW);
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    delay(3);
    #ifdef OUTPUT_READABLE_ACCELGYRO
        Serial.print(ax); Serial.print("\t");
        Serial.print(ay); Serial.print("\t");
        Serial.print(az); Serial.print("\t");
        Serial.print(gx); Serial.print("\t");
        Serial.print(gy); Serial.print("\t");
        Serial.print(gz); Serial.print("\t");
    #endif

    
    digitalWrite(ADO2_PIN, HIGH);
    digitalWrite(ADO3_PIN, LOW);
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    delay(3);
    #ifdef OUTPUT_READABLE_ACCELGYRO
        Serial.print(ax); Serial.print("\t");
        Serial.print(ay); Serial.print("\t");
        Serial.print(az); Serial.print("\t");
        Serial.print(gx); Serial.print("\t");
        Serial.print(gy); Serial.print("\t");
        Serial.print(gz); Serial.print("\t");
    #endif



    digitalWrite(ADO3_PIN, HIGH);
    digitalWrite(ADO4_PIN, LOW);
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    delay(3);
    #ifdef OUTPUT_READABLE_ACCELGYRO
        Serial.print(ax); Serial.print("\t");
        Serial.print(ay); Serial.print("\t");
        Serial.print(az); Serial.print("\t");
        Serial.print(gx); Serial.print("\t");
        Serial.print(gy); Serial.print("\t");
        Serial.println(gz);
    #endif

    digitalWrite(ADO4_PIN, HIGH);
    digitalWrite(ADO1_PIN, LOW);
}




/*
    #ifdef OUTPUT_READABLE_ACCELGYRO
        Serial.print(ax); Serial.print("\t");
        Serial.print(ay); Serial.print("\t");
        Serial.print(az); Serial.print("\t");
        Serial.print(gx); Serial.print("\t");
        Serial.print(gy); Serial.print("\t");
        Serial.print(gz); Serial.print("\t");
    #endif
  
  
  #ifdef OUTPUT_BINARY_ACCELGYRO
    Serial.write((uint8_t)(ax >> 8)); Serial.write((uint8_t)(ax & 0xFF));
    Serial.write((uint8_t)(ay >> 8)); Serial.write((uint8_t)(ay & 0xFF));
    Serial.write((uint8_t)(az >> 8)); Serial.write((uint8_t)(az & 0xFF));
    Serial.write((uint8_t)(gx >> 8)); Serial.write((uint8_t)(gx & 0xFF));
    Serial.write((uint8_t)(gy >> 8)); Serial.write((uint8_t)(gy & 0xFF));
    Serial.write((uint8_t)(gz >> 8)); Serial.write((uint8_t)(gz & 0xFF));
  #endif
*/
