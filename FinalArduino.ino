#include <SR04.h>
//#include "IRremoteNew.h"
#include <pitches.h>
SR04 ultraSonic = SR04(8,  7); // Sets up the UltraSonic using 12 for ECHO and 11 for TRIG
long dist;

//for the lcd 9=13, 8=9, 7=8, 6=7, 2=1, 3=3
/*-----( Declare objects )-----
Both of these are coming from the library that we included up above.*/
//IRrecv irrecv(receiver);     // create instance of 'irrecv'
//decode_results results;      // create instance of 'decode_results'


const int buzzerPin=10;

int data;
int Cdist;
int songSize = 2; //Stores the number of items in the melody.
int melody[] = {NOTE_C4, NOTE_C4}; //Stores the notes for the first melody. (The "0" is a rest)
int duration[] = {80,20}; // Stores the length of each note in the first melody

void setup() {
  // put your setup code here, to run once:
  //irrecv.enableIRIn(); // Start the receiver
  Serial.begin(9600);
}

void loop() {
  Serial.println(data);
  data = Serial.read();
  dist = ultraSonic.Distance();
  if (data == '1'){
    Cdist = 0;
  }
   else if (data == '0'){
    Cdist = 209;
   }

  if(dist < Cdist){
    for(int i = 0; i < songSize; i++){
      int d = 10000/duration[i];
       tone(buzzerPin, melody[i], d);
      int pause = d * 1.30;
      delay(pause);
      noTone(buzzerPin);
    delay(5);
    }
  }
}
