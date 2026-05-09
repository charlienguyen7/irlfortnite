/*
Push this code onto your ESP32
*/

#include "BluetoothSerial.h"

BluetoothSerial SerialBT;

const int triggerPin = 4; // GPIO pin connected to the trigger (button)
const int joyXPin = 32; // GPIO pin connected to Vx of joystick (ADC)
const int joyYPin = 33; // GPIO pin connected to Vy of joystick (ADC)

uint8_t trigger;
uint8_t joystickX;
uint8_t joystickY;

void setup() {
    Serial.begin(115200);
    SerialBT.begin("ESP32_Shotgun"); // Bluetooth device name
    Serial.println("Pair with Bluetooth to Activate");

    pinMode(triggerPin, INPUT); // Set trigger pin as input
    pinMode(joystickX, INPUT);
    pinMode(joystickY, INPUT);
}

void loop() {
    trigger = digitalRead(triggerPin);
    joystickX = (analogRead(joyXPin) >> 4); // right shift 16-bits since gamepad input is 2^8 and ADC 2^12
    joystickY = (analogRead(joyYPin) >> 4); // right shift 16-bits since gamepad input is 2^8 and ADC 2^12
    SerialBT.println()
}




