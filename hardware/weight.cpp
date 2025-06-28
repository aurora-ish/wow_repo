#include "HX711.h"

const int DOUT = 3;
const int SCK = 2;

HX711 scale;

void setup() {
  Serial.begin(9600);
  scale.begin(DOUT, SCK);

  while (!scale.is_ready()) {
    Serial.println("Waiting for HX711...");
    delay(500);
  }

  Serial.println("HX711 ready.");
}

void loop() {
  if (scale.is_ready()) {
    long raw_value = scale.read();  
    float voltage = (raw_value / 8388607.0) * 1000.0;  

    Serial.print("Raw ADC: ");
    Serial.print(raw_value);
    Serial.print(" | Voltage: ");
    Serial.println(voltage, 6); 
  } else {
    Serial.println("HX711 not ready.");
  }

  delay(500);
}
