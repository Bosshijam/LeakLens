/*
 * LeakLens - local oscillator + reference emitter
 * ------------------------------------------------
 * Standalone sketch for a spare Arduino Uno / Nano (ATmega328P @ 16 MHz).
 * Use this on DAY 1, before the UNO Q arrives.
 *
 * Two independent outputs:
 *
 *   PIN 11 (OC2A)  38 kHz  -> local oscillator for the mixer
 *   PIN  9 (OC1A)  40 kHz  -> drives an MA40S4S transmitter, your reference
 *                             source for beam characterisation and for testing
 *                             the receive chain without needing compressed air
 *
 * WHY TWO SEPARATE TIMERS
 * Timer0 runs millis()/delay() - leave it alone.
 * Timer1 is 16-bit, good for the precise 40 kHz emitter.
 * Timer2 is 8-bit, adequate for the LO.
 * The main firmware (leaklens_mcu.ino) needs Timer1 for the Servo library, which
 * is why the LO lives on Timer2 there too. Keep that consistent.
 *
 * SERIAL COMMANDS (250000 baud)
 *   L <hz>     set LO frequency, e.g.  L 38000
 *   E <hz>     set emitter frequency,  e.g.  E 40000
 *   E 0        emitter off
 *   S          report current settings
 *
 * TUNING TIP
 * Sweep the LO from 36000 to 40000 while watching tools/capture.py. Your
 * transducer's true resonance is rarely exactly 40.0 kHz - find the LO setting
 * that puts the difference tone where you want it and gives the biggest
 * response, then hard code that value.
 */

const uint8_t PIN_LO      = 11;   // OC2A
const uint8_t PIN_EMITTER = 9;    // OC1A

uint32_t loHz = 38000;
uint32_t emHz = 40000;

/* ---------------------------------------------------------------- Timer2
 * CTC mode, toggle OC2A.
 *   f = F_CPU / (2 * N * (1 + OCR2A))
 * With N = 1 (no prescale) and OCR2A 8-bit, the reachable range is roughly
 * 31.4 kHz (OCR=255) to 8 MHz. 38 kHz -> OCR2A = 209.
 */
bool setLO(uint32_t hz) {
  if (hz == 0) {                     // off
    TCCR2A = 0; TCCR2B = 0;
    pinMode(PIN_LO, OUTPUT); digitalWrite(PIN_LO, LOW);
    loHz = 0;
    return true;
  }
  uint32_t ocr = (F_CPU / (2UL * hz));
  if (ocr < 1 || ocr > 256) return false;
  ocr -= 1;

  pinMode(PIN_LO, OUTPUT);
  TCCR2A = _BV(COM2A0) | _BV(WGM21);   // toggle OC2A on compare, CTC
  TCCR2B = _BV(CS20);                  // no prescaler
  OCR2A  = (uint8_t)ocr;
  TCNT2  = 0;
  loHz = F_CPU / (2UL * (ocr + 1));     // actual, not requested
  return true;
}

/* ---------------------------------------------------------------- Timer1
 * CTC mode, toggle OC1A. 16-bit so the frequency resolution is much finer.
 * 40 kHz -> OCR1A = 199 exactly, which is a nice round 40.000 kHz.
 *
 * NOTE ON DRIVE LEVEL
 * A single 5 V pin gives ~5 Vpp across the MA40S4S, which is enough for bench
 * work at a metre or two. If you need more range, drive it push-pull from two
 * inverted outputs, or through a couple of 74HC04 sections in parallel, for
 * ~10 Vpp. The transducer is rated well above this.
 */
bool setEmitter(uint32_t hz) {
  if (hz == 0) {
    TCCR1A = 0; TCCR1B = 0;
    pinMode(PIN_EMITTER, OUTPUT); digitalWrite(PIN_EMITTER, LOW);
    emHz = 0;
    return true;
  }
  uint32_t ocr = (F_CPU / (2UL * hz));
  if (ocr < 1 || ocr > 65536) return false;
  ocr -= 1;

  pinMode(PIN_EMITTER, OUTPUT);
  TCCR1A = _BV(COM1A0);                        // toggle OC1A
  TCCR1B = _BV(WGM12) | _BV(CS10);             // CTC, no prescaler
  OCR1A  = (uint16_t)ocr;
  TCNT1  = 0;
  emHz = F_CPU / (2UL * (ocr + 1));
  return true;
}

void report() {
  Serial.print(F("LO ")); Serial.print(loHz);
  Serial.print(F(" Hz  EMITTER ")); Serial.print(emHz);
  Serial.println(F(" Hz"));
}

void setup() {
  Serial.begin(250000);          // 0% baud error at 16 MHz, unlike 115200
  setLO(loHz);
  setEmitter(emHz);
  Serial.println(F("LeakLens LO generator ready"));
  report();
}

void loop() {
  if (!Serial.available()) return;

  char cmd = Serial.read();
  if (cmd == '\n' || cmd == '\r') return;

  uint32_t val = Serial.parseInt();
  while (Serial.available() && (Serial.peek() == '\n' || Serial.peek() == '\r'))
    Serial.read();

  switch (cmd) {
    case 'L': case 'l':
      Serial.println(setLO(val) ? F("OK") : F("ERR range"));
      report();
      break;
    case 'E': case 'e':
      Serial.println(setEmitter(val) ? F("OK") : F("ERR range"));
      report();
      break;
    case 'S': case 's':
      report();
      break;
    default:
      Serial.println(F("ERR unknown"));
  }
}
