#include "Arduino_BHY2.h"
#include "ArduinoBLE.h"

int all_readings[2500];

void setup() {
  Serial.begin(115200);
  while(!Serial);
  all_readings[0] = 0;
  while(!BHY2.begin());
  while(!BLE.begin());
}

void loop() {
  BHY2.update(); 
  Serial.println("We are working fine!");
  delay(1000);
}