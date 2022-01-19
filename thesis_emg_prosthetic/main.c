#include <msp430G2553.h>


// Attempting to use Pin 2.1 and 2.4 on Timer 1
// 2020_04_24 00:38 Both pins work - alternating rotations!
// 2020_05_03 3:33 pm Oh my gosh it works! Slowly contracts finger... now must allow opening

static const int left = 300;//350; // finger open
static const int right = 2500;//2600; // finger contracted
const int range = right - left;//2600-350;
volatile unsigned int i;
volatile unsigned int j;
volatile int opening = 1;

void setTimer(){
    //***********Start code, set to 1 MHz (from examples)*
    DCOCTL = 0x00;
    BCSCTL1 = CALBC1_1MHZ; /* Set DCO to 1MHz */
    DCOCTL = CALDCO_1MHZ;
    /* Basic Clock System Control 1
    * XT2OFF -- Disable XT2CLK
    * ~XTS -- Low Frequency
    * DIVA_0 -- Divide by 1 */
    BCSCTL1 |= XT2OFF + DIVA_0;
    /* Basic Clock System Control 3
    * XT2S_0 -- 0.4 - 1 MHz
    * LFXT1S_2 -- If XTS = 0, XT1 = VLOCLK ; If XTS = 1, XT1 = 3 - 16-MHz crystal or resonator
    * XCAP_1 -- ~6 pF */
    BCSCTL3 = XT2S_0 + LFXT1S_2 + XCAP_1;
    //***********End code, set to 1 MHz*************
}

void correctRange(int sel, int val){
    if (sel == 1){TA1CCR1 = val;}
    else {TA1CCR2 = val;}
}

int checkRangeConform(int sel){
    int t_check;
    if (sel == 1){t_check = TA1CCR1;}
    else {t_check = TA1CCR2;}
    if (t_check < left){
        correctRange(sel, left);
        return 0;}
    if (t_check > right){
        correctRange(sel, right);
        return 0;}
    return 1;
}

void waitX(int ms){
    volatile int waitclk;
    for(waitclk=ms;waitclk>0;waitclk--);
}
//void countround30Degrees(int sel, int delta){
//    if (checkRangeConform(sel) == 0){return;}
//    for (i = 0; i<(range/6); i++){
//        if (sel == 1){TA1CCR1 += delta;}  // TA1.1 for pins 2.1 and 2.2
//        else {TA1CCR2 += delta;} // TA1.2 for pins 2.4 and 2.5
//        waitX(80);
//    }
//    if (checkRangeConform(sel) == 0){
//        opening = opening*(-1);
//        //waitX(5000);
//        return;}
//}
//

void countround30Degrees(int delta){
    if (checkRangeConform(1) == 0){return;}
    for (i = 0; i<(range/6); i++){
        TA1CCR1 += delta;  // TA1.1 for pins 2.1 and 2.2
        TA1CCR2 += delta; // TA1.2 for pins 2.4 and 2.5
        waitX(80);
    }
    if (checkRangeConform(1) == 0){
        opening = opening*(-1);
        waitX(5000);
        return;}
}

void setupTimersAndStuff(){
    setTimer();
    WDTCTL = WDTPW + WDTHOLD; //Disable the Watchdog timer for our convenience.
    P2DIR |= BIT1 | BIT4; //Set pin 1.2 to the output direction.
    P2SEL |= BIT1 | BIT4; //Select pin 1.2 as our PWM output.
    TA1CCR0 = 20000-1; //Set the period in the Timer A0 Capture/Compare 0 register to 1000 us.
    TA1CCTL1 = OUTMOD_7; // for Control 1 (TA1.1)
    TA1CCTL2 = OUTMOD_7; // for Control 2 (TA1.2)
    TA1CTL = TASSEL_2 + MC_1; //TASSEL_2 selects SMCLK as the clock source, and MC_1 tells it to count up to the value in TA0CCR0.

    // Timer 1 is set to a PWM of 50 Hz using TA1CCR0
    // TA1CCR1 controls P2.1
    // TA1CCR2 controls P2.4
    TA1CCR1 = left;
    TA1CCR2 = left;
    waitX(5000);
}

void setupADC(){
    // Setup A/D
    ADC10CTL1 = INCH_3;        //Set to channel 3 (Pin 1.3) - underscore indicates decimal numbering system
    ADC10CTL0 = ADC10ON | ENC; //Turn ADC on, enable conversions
}

int getADC(){
    int adval;
    ADC10CTL0 |= ADC10SC;  // Start conversion
    while (ADC10CTL1 & ADC10BUSY); // Wait for conversion to finish
    adval = ADC10MEM;      // Get A/D conversion result
    if ((adval>1024) || (adval < 0)){
        adval = 1010;
    }
    return adval;
}

int emgFlex(){
    volatile int emg;
    emg = getADC();
    if (emg > 600){
        if (opening >0){return 1;}
        else {return -1;}
    }
    return 0;
}

int main(void) {
    setupTimersAndStuff();
    setupADC();
    volatile int act;
    while (1){
        act = emgFlex();
        countround30Degrees(act);
        //countround30Degrees(2,act);
        //-1 -> open, 1 -> contract, 0 -> nothing
    }
    //__bis_SR_register(LPM0_bits); //Switch to low power mode 0.
}
