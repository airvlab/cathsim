#include <MultiStepper.h>

#include <AccelStepper.h> // you need to install library
#include <Arduino.h>

#define ulong unsigned long

#define enable_pin 8

// Define steppers and the pins they will use
AccelStepper stepper1(AccelStepper::FULL2WIRE, 2, 5);   // X Fast, top right ?? top left
AccelStepper stepper2(AccelStepper::FULL2WIRE, 3, 6);   // Y Slow, top left ??top right
AccelStepper stepper3(AccelStepper::FULL2WIRE, 4, 7);   // Z Fast, bottom left
AccelStepper stepper4(AccelStepper::FULL2WIRE, 12, 13); // A Slow, bottom right

int incomingByte = 0; // for incoming serial data

volatile bool data_ready = false;
volatile bool relative = true;
volatile long steps[4];
volatile bool set_enable = false;

void setup()
{
  pinMode(enable_pin, OUTPUT);    // sets the digital pin 13 as output?? pin 8
  digitalWrite(enable_pin, HIGH); // Low equals motors enabled; High equals motors disabled

  pinMode(LED_BUILTIN, OUTPUT);

  stepper1.setMaxSpeed(10000.0);
  stepper1.setAcceleration(5000.0);

  stepper2.setMaxSpeed(10000.0);
  stepper2.setAcceleration(5000.0);

  stepper3.setMaxSpeed(10000.0);
  stepper3.setAcceleration(5000.0);

  stepper4.setMaxSpeed(10000.0);
  stepper4.setAcceleration(5000.0);

  Serial.begin(115200);          // for serial input
  digitalWrite(enable_pin, LOW); // Low equals motor enabled
}
void loop()
{
  serialEvent();
  if (data_ready) // If you use serial communication
  {
    data_ready = false;
    if (relative)
    {
      stepper3.move(steps[2]); // linear
      stepper4.move(steps[3]); // rotation
    }
    else
    {
      stepper3.moveTo(steps[2]); // linear
      stepper4.moveTo(steps[3]); // rotation
    }
    // stepper3.moveTo(3000);  // linear // each rotation is 8mm, and one rotation is 800 steps
    // stepper4.moveTo(400);  // rotation, each 360rotation is 800 steps
  }
  // This make the motors move and must be call continuously
  stepper1.run();
  stepper2.run();
  stepper3.run();
  stepper4.run();
}

// to be used if you want to control the motors from your pc via a serial communication format is one data frame starting with 0x81 0x88 motor1 motor2 motor3  motor4, each motor is signed long on 4 bytes
void serialEvent()
{
  unsigned char data[19];

  if (Serial.available())
  {
    // get the new byte:
    Serial.readBytes(data, 19); // read data from serial
    // test---------------------------
    // Serial.println("Received in Arduino: ");
    // Serial.println(data[0]);
    // test end-----------------------
    set_enable = data[0] & 0x01;

    int j = 2;
    for (int i = 0; i < 4; i++)
    {
      steps[i] = ((ulong)(data[j++]) << 24) | ((ulong)(data[j++]) << 16) | ((ulong)(data[j++]) << 8) | data[j++];
    }

    if ((data[0] == 0x81) || (data[0] == 0x80))
      if (data[1] == 0x88)
        data_ready = true;
    if (data[18] == 0x81)
      relative = false;
  }
}