/*
Push this code onto your ESP32
*/

#include "BluetoothSerial.h"
#define OUTPUTSIZE 3

BluetoothSerial SerialBT;

const int triggerPin = 4; // GPIO pin connected to the trigger (button)
const int joyXPin = 32; // GPIO pin connected to Vx of joystick (ADC)
const int joyYPin = 33; // GPIO pin connected to Vy of joystick (ADC)

uint8_t trigger;
uint8_t joystickX;
uint8_t joystickY;

uint8_t output[OUTPUTSIZE]; // array containing outputs to be sent through Bluetooth

void setup() {
    Serial.begin(115200);
    SerialBT.begin("ESP32_Shotgun"); // Bluetooth device name
    Serial.println("Pair with Bluetooth to Activate");

    pinMode(triggerPin, INPUT); // Set trigger pin as input (I'm using external pulldown resisitor; otherwise use INPUT_PULLDOWN)
    pinMode(joyXPin, INPUT); // Set joystick Vx pin as input
    pinMode(joyYPin, INPUT); // Set joystick Vy pin as input
}

void loop() {
    output[0] = digitalRead(triggerPin); // read if button is being pressed    
    output[1] = (analogRead(joyXPin) >> 4); // right shift 4-bits since gamepad input is 2^8 and ADC 2^12
    output[2] = (analogRead(joyYPin) >> 4); // right shift 4-bits since gamepad input is 2^8 and ADC 2^12
    SerialBT.write(output, OUTPUTSIZE); // output array containing all data elements
    delay(20);
}




