#include <SoftwareSerial.h>
#include <TinyGPS++.h>

static const int RXPin = 4, TXPin = 3; 
static const uint32_t GPSBaud = 9600;

TinyGPSPlus gps;
SoftwareSerial ss(RXPin, TXPin);

void setup() {
  Serial.begin(9600);     
  ss.begin(GPSBaud);    
  Serial.println("Initializing GPS...");
}

void loop() {
  while (ss.available() > 0) {
    gps.encode(ss.read());
    
    if (gps .location.isUpdated()) {
      Serial.print("Latitude: ");
      Serial.println(gps.location.lat(), 6);
      Serial.print("Longitude: ");
      Serial.println(gps.location.lng(), 6);
      Serial.print("Satellites: ");
      Serial.println(gps.satellites.value());
      Serial.print("Speed (km/h): ");
      Serial.println(gps.speed.kmph());
      Serial.println("------");
    }
  }
}
