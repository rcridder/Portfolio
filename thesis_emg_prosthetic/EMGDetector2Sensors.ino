#include <Adafruit_CircuitPlayground.h>
#include <TimerOne.h>
#include <Filters.h>

#define measureDuty 10 // 100 microseconds
#define datapin 6

uint16_t val;
float voltage;
FilterOnePole LPFilter;
unsigned long timetime;
int pinout = 0;
int binary = 0;

void interruptSetup(){
  // interrupt
  Timer1.initialize(measureDuty); // in microseconds
  Timer1.attachInterrupt(measureEMG, measureDuty);
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(38400);
  CircuitPlayground.begin();
  interruptSetup();
  
  pinMode(A9, OUTPUT);
  pinMode(13, OUTPUT);
}

float filterEMG(float voltage){
  float filterFrequency = 60.0;
  FilterOnePole lowpassFilter(LOWPASS, filterFrequency-2);
  FilterOnePole highpassFilter(HIGHPASS, filterFrequency+2);

  highpassFilter.input(voltage);
  lowpassFilter.input(highpassFilter.output());
  return lowpassFilter.output();
}

void measureEMG(){
//  //timing
//  timetime = millis();
//  Serial.print(timetime);
//  Serial.print("\t");

  // readEMG
  val = analogRead(A7);
  voltage = ((1.0*val)/1023)*3.3;
  Serial.print(voltage);
  Serial.print("\t");

//  val = analogRead(A11);
//  voltage = ((1.0*val)/1023)*3.3;
//  Serial.print(voltage);
//  Serial.print("\t");
  
  //filter EMG
  float filtered = voltage - 1.65;//filterEMG(voltage);
  Serial.print(filtered);
  Serial.print('\t');

  // threshold EMG
  if (abs(filtered) > .1){
    digitalWrite(A9, HIGH); binary = 1; // set A9 High as EMG flag
    digitalWrite(13, HIGH);
    pinout = 0; // reset counter
  }
  if (pinout == 100){digitalWrite(A9, LOW); binary = 0; // reset A9 to low
  digitalWrite(13, LOW);
  }else{pinout++;} // update counter
  
  Serial.print(binary); // indicates status of digital output A9
  Serial.print("\t");
  Serial.println("");
}

void loop() {
  // put your main code here, to run repeatedly:
  // emg not needed in the loop because it is read using interrupts
}
