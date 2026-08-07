/*
 * LeakLens - MCU firmware
 * -----------------------
 * Implements the wire protocol that leaklens/device.py::SerialDevice speaks.
 *
 * RUN THIS ON A SPARE ARDUINO NANO TODAY. The same protocol, and most of this
 * code, moves to the UNO Q's STM32U585 when it arrives - see the PORTING notes
 * at the bottom. That is the whole point: your Linux-side application never
 * knows or cares which board is underneath.
 *
 * WIRING (Nano / Uno)
 *   A0        mixer output, biased to VCC/2   <- the signal
 *   D11 (OC2A) 38 kHz local oscillator out    -> mixer LO input
 *   D5        pan servo
 *   D6        tilt servo
 *   D7        laser (through a transistor, not directly)
 *   A4/A5     I2C for VL53L1X, optional
 *
 * !!! THE TIMER TRAP !!!
 * The Servo library takes Timer1 on AVR. If you also try to generate the LO on
 * Timer1 the servos jitter wildly and the LO frequency drifts. It looks like a
 * mechanical fault and it is not. LO goes on TIMER2. This costs people a full
 * day; it is written here so it does not cost you one.
 *
 * PROTOCOL (250000 baud, newline terminated, host always initiates)
 *   P              -> "P leaklens <ver>"
 *   A <az> <el>    -> "OK"                 move, block until settled
 *   E <dwell_ms>   -> "E <rms>"            RMS of the AC signal over dwell
 *   W <ms>         -> "W <sr> <n>" + CSV   waveform, capped at BUF_LEN
 *   D              -> "D <metres>"         -1 if no ranging sensor
 *   L <0|1>        -> "OK"
 */

#include <Servo.h>

#define FW_VERSION 1

const uint8_t PIN_SIGNAL = A0;
const uint8_t PIN_LO     = 11;    // OC2A - Timer2, NOT Timer1
const uint8_t PIN_PAN    = 5;
const uint8_t PIN_TILT   = 6;
const uint8_t PIN_LASER  = 7;

/* Mechanical limits and calibration. Measure these on your own head - servo
 * horns never sit exactly where you think. */
const float AZ_MIN = -40.0, AZ_MAX = 40.0;
const float EL_MIN = -30.0, EL_MAX = 30.0;
const float AZ_CENTER_US = 1500.0, EL_CENTER_US = 1500.0;
const float US_PER_DEG   = 10.0;   // ~10 us per degree for a typical hobby servo

/* Buffer: 512 int16 = 1 kB of the Nano's 2 kB SRAM. Do not raise this on AVR.
 * On the UNO Q's STM32 (786 kB SRAM) take it to 16384 and capture a full
 * second in one shot. */
#define BUF_LEN 512
int16_t buf[BUF_LEN];

Servo panServo, tiltServo;
float curAz = 0, curEl = 0;
uint16_t dcBias = 512;             // measured at boot

/* ------------------------------------------------------------------ ADC
 * Default analogRead() is ~9.6 kS/s because the prescaler is 128. Our
 * heterodyned signal sits around 2 kHz, so we want at least 8-16 kS/s.
 * Prescaler 32 -> ADC clock 500 kHz -> ~38 kS/s. Accuracy degrades slightly
 * above 200 kHz ADC clock but is entirely adequate for relative measurements.
 */
void fastADC() {
  ADCSRA = (ADCSRA & ~0x07) | 0x05;   // prescaler 32
}

uint16_t readRaw() {
  return analogRead(PIN_SIGNAL);
}

void measureBias() {
  uint32_t acc = 0;
  for (uint16_t i = 0; i < 1024; i++) acc += readRaw();
  dcBias = acc / 1024;
}

/* ------------------------------------------------------------------- LO */
void startLO(uint32_t hz) {
  uint32_t ocr = (F_CPU / (2UL * hz));
  if (ocr < 1 || ocr > 256) return;
  ocr -= 1;
  pinMode(PIN_LO, OUTPUT);
  TCCR2A = _BV(COM2A0) | _BV(WGM21);   // toggle OC2A, CTC
  TCCR2B = _BV(CS20);                  // no prescaler
  OCR2A  = (uint8_t)ocr;
}

/* ---------------------------------------------------------------- motion
 * Settling time is the hidden cost of a raster scan. Too short and you sample
 * while the head is still ringing, smearing the image. Too long and a 600-point
 * scan takes minutes. Measure yours: command a 10 degree step, watch the signal
 * on a scope, see when it stops moving. 120 ms base is a safe starting point
 * for an MG996R carrying a dish.
 */
void moveTo(float az, float el) {
  az = constrain(az, AZ_MIN, AZ_MAX);
  el = constrain(el, EL_MIN, EL_MAX);

  float travel = max(fabs(az - curAz), fabs(el - curEl));

  panServo.writeMicroseconds((int)(AZ_CENTER_US + az * US_PER_DEG));
  tiltServo.writeMicroseconds((int)(EL_CENTER_US + el * US_PER_DEG));

  uint16_t settle = 60 + (uint16_t)(travel * 6.0);   // ms
  if (settle > 400) settle = 400;
  delay(settle);

  curAz = az; curEl = el;
}

/* -------------------------------------------------------------- envelope
 * RMS of the AC component over the dwell window. This single number is what
 * the raster scan collects at every grid point.
 */
float readEnvelope(uint16_t dwellMs) {
  uint32_t t0 = millis();
  uint32_t n = 0;
  float acc = 0;
  while ((millis() - t0) < dwellMs) {
    int32_t v = (int32_t)readRaw() - (int32_t)dcBias;
    acc += (float)(v * v);
    n++;
  }
  if (n == 0) return 0.0f;
  return sqrt(acc / (float)n);
}

/* -------------------------------------------------------------- waveform
 * Uniformly-timed capture into RAM, then CSV out. CSV is slow but it is only
 * requested ONCE per scan (at the peak, for classification), so it costs about
 * 100 ms total. Keeping it human-readable is worth far more during bring-up
 * than the bandwidth saved by a binary format.
 */
void captureWaveform(uint16_t ms) {
  const uint32_t sr = 8000;                      // plenty for a ~2 kHz signal
  uint16_t want = (uint32_t)ms * sr / 1000UL;
  if (want > BUF_LEN) want = BUF_LEN;

  const uint32_t periodUs = 1000000UL / sr;
  uint32_t next = micros();
  for (uint16_t i = 0; i < want; i++) {
    while ((int32_t)(micros() - next) < 0) { /* spin for even spacing */ }
    buf[i] = (int16_t)((int32_t)readRaw() - (int32_t)dcBias);
    next += periodUs;
  }

  Serial.print(F("W ")); Serial.print(sr);
  Serial.print(' ');     Serial.println(want);
  for (uint16_t i = 0; i < want; i++) {
    Serial.print(buf[i]);
    if ((i % 32) == 31 || i == want - 1) Serial.println();
    else Serial.print(',');
  }
}

/* -------------------------------------------------------------- distance
 * Stub. Fit a VL53L1X and return real metres - see the Pololu VL53L1X library.
 * Returning -1 makes the host fall back to a configured nominal distance, so
 * the system works without it.
 */
float readDistance() {
  return -1.0f;
}

/* ------------------------------------------------------------------ main */
void setup() {
  Serial.begin(250000);
  pinMode(PIN_LASER, OUTPUT);
  digitalWrite(PIN_LASER, LOW);

  panServo.attach(PIN_PAN);
  tiltServo.attach(PIN_TILT);

  fastADC();
  startLO(38000);
  moveTo(0, 0);
  delay(300);
  measureBias();

  Serial.print(F("P leaklens ")); Serial.println(FW_VERSION);
}

void loop() {
  if (!Serial.available()) return;
  char cmd = Serial.read();

  switch (cmd) {
    case 'P': {
      Serial.print(F("P leaklens ")); Serial.println(FW_VERSION);
      break;
    }
    case 'A': {
      float az = Serial.parseFloat();
      float el = Serial.parseFloat();
      moveTo(az, el);
      Serial.println(F("OK"));
      break;
    }
    case 'E': {
      long d = Serial.parseInt();
      if (d <= 0) d = 40;
      Serial.print(F("E "));
      Serial.println(readEnvelope((uint16_t)d), 4);
      break;
    }
    case 'W': {
      long ms = Serial.parseInt();
      if (ms <= 0) ms = 60;
      captureWaveform((uint16_t)ms);
      break;
    }
    case 'D': {
      Serial.print(F("D "));
      Serial.println(readDistance(), 3);
      break;
    }
    case 'L': {
      long on = Serial.parseInt();
      digitalWrite(PIN_LASER, on ? HIGH : LOW);
      Serial.println(F("OK"));
      break;
    }
    case '\n': case '\r':
      break;
    default:
      Serial.println(F("ERR"));
  }
}

/* =====================================================================
 * PORTING TO THE UNO Q  (day 8-10)
 *
 * 1. TIMERS. The STM32U585 has no AVR timer registers. Replace startLO() with
 *    an STM32 timer in PWM mode at 38 kHz, 50% duty. The Servo library should
 *    work through the Arduino core; if it fights the LO timer, use a different
 *    timer instance - the U585 has plenty, unlike the ATmega.
 *
 * 2. ADC. Replace the analogRead() spin loop with DMA-driven sampling on a
 *    timer trigger. This is the one genuinely new piece of work and the reason
 *    the plan allocates days 9-10 to it. Target 16 kS/s continuous into a
 *    double buffer. Verify with the MA40S4S emitter at a known level before
 *    trusting any of it.
 *
 * 3. BUFFER. Raise BUF_LEN to 16384. You have 786 kB of SRAM, not 2 kB.
 *
 * 4. TRANSPORT. Keep this ASCII protocol over USB CDC initially - it works and
 *    it is debuggable. Migrate to the native Bridge RPC only once everything
 *    else is proven, and only by filling in UnoQBridgeDevice in device.py.
 *    Nothing above that class needs to change.
 *
 * Do NOT do all four at once. Get the ADC working with the ASCII protocol
 * first, confirm the full scan pipeline still runs, and only then consider the
 * native Bridge.
 * ===================================================================== */
