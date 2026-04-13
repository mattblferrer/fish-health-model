/*
 * ESP32 Audio Band-Pass Filter with SD Card Recorder + I2C LCD
 * =============================================================
 * Captures audio via analog input (piezo / mic),
 * applies a configurable band-pass filter,
 * saves filtered audio to a WAV file on SD card,
 * and displays recording status on a 16x2 I2C LCD.
 *
 * Hardware connections:
 *   I2C LCD (16x2 or 20x4 with PCF8574 backpack)
 *     VCC -> 5V
 *     GND -> GND
 *     SDA -> GPIO 25
 *     SCL -> GPIO 26
 *
 *   SD Card Module (SPI)
 *     MOSI -> GPIO 23
 *     MISO -> GPIO 19
 *     SCK  -> GPIO 18
 *     CS   -> GPIO 5
 *     VCC  -> 3.3V / 5V (per module spec)
 *     GND  -> GND
 *
 *   Piezo / analog mic input -> GPIO 34
 *
 * Required libraries (install via Library Manager):
 *   - LiquidCrystal I2C  (by Frank de Brabander)
 *   - SD (built-in with ESP32 Arduino core)
 */

#include <Arduino.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <SD.h>
#include <SPI.h>
#include <math.h>
#include <esp_adc/adc_continuous.h>
#include <esp_adc/adc_cali.h>
#include <esp_adc/adc_cali_scheme.h>

// ─────────────────────────────────────────────
//  USER CONFIGURATION
// ─────────────────────────────────────────────

// ADC input
#define ADC_CHANNEL_NUM  ADC_CHANNEL_6   // GPIO34
#define ADC_UNIT_NUM     ADC_UNIT_1

// Recording duration (seconds). Set 0 for button-controlled recording.
#define RECORD_SECONDS      10

// Sample rate in Hz — minimum 20000 for ESP32 ADC continuous mode
#define SAMPLE_RATE         32000

#define BITS_PER_SAMPLE     16
#define CHANNELS            1

// Band-pass filter cutoffs. Set either to 0 to disable that stage.
#define FILTER_LOW_HZ       0
#define FILTER_HIGH_HZ      1000

// Cascaded biquad stages (each adds 12 dB/oct roll-off)
#define HP_STAGES           3
#define LP_STAGES           3

// Software gain applied AFTER filtering
#define SOFTWARE_GAIN       1.0f

// ADC bias midpoint in millivolts (1650 mV for a centred 3.3V divider)
#define BIAS_MV             1650.0f

// Output filename base (/rec_0001.wav, /rec_0002.wav …)
#define FILENAME_BASE       "/rec_"

// Counter file — stores the last used file number so nextFilename()
// never has to scan the SD card.
#define COUNTER_FILE        "/recnum.txt"

// ── Pin assignments ───────────────────────────
#define SD_CS_PIN           5
#define STOP_BUTTON_PIN     0    // GPIO0 = BOOT button

// ── LCD configuration ─────────────────────────
// Common I2C addresses for PCF8574 backpacks: 0x27 or 0x3F
// If the display stays blank, try the other address.
#define LCD_I2C_ADDRESS     0x27
#define LCD_COLS            16
#define LCD_ROWS            2

// ── I2C pins ──────────────────────────────────
#define I2C_SDA_PIN         25
#define I2C_SCL_PIN         26

// ─────────────────────────────────────────────
//  INTERNAL CONSTANTS
// ─────────────────────────────────────────────

#define BYTES_PER_SAMPLE    (BITS_PER_SAMPLE / 8)
#define WAV_HEADER_SIZE     44
#define READ_BUF_SAMPLES    1024
#define CONV_FRAME_SIZE     (READ_BUF_SAMPLES * SOC_ADC_DIGI_RESULT_BYTES)

// ─────────────────────────────────────────────
//  BIQUAD FILTER STRUCTS
// ─────────────────────────────────────────────

struct BiquadCoeff {
  double b0, b1, b2;
  double a1, a2;
};

struct BiquadState {
  double x1, x2;
  double y1, y2;
};

// ─────────────────────────────────────────────
//  GLOBALS
// ─────────────────────────────────────────────

// LCD
static LiquidCrystal_I2C lcd(LCD_I2C_ADDRESS, LCD_COLS, LCD_ROWS);

// Recording
static File     g_wavFile;
static uint32_t g_dataBytesWritten = 0;

// Filters
static BiquadCoeff g_hpCoeff, g_lpCoeff;
static BiquadState g_hpState[HP_STAGES], g_lpState[LP_STAGES];
static bool        g_hpEnabled = false;
static bool        g_lpEnabled = false;

// ADC — calibration handle created once in setup(), never recreated
static adc_continuous_handle_t adc_cont_handle = NULL;
static adc_cali_handle_t       adc_cali_handle = NULL;

// SD state — only remount when the card was previously unmounted
static bool g_sdMounted  = false;

// File counter — incremented each recording, persisted to COUNTER_FILE
static int  g_lastFileNum = 0;

// ═════════════════════════════════════════════
//  LCD HELPERS
// ═════════════════════════════════════════════

void lcdShow(const char *line1, const char *line2 = "") {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(line1);
  if (LCD_ROWS >= 2) {
    lcd.setCursor(0, 1);
    lcd.print(line2);
  }
}

void lcdLine2(const char *text) {
  if (LCD_ROWS < 2) return;
  lcd.setCursor(0, 1);
  char padded[LCD_COLS + 1];
  snprintf(padded, sizeof(padded), "%-*s", LCD_COLS, text);
  lcd.print(padded);
}

// ═════════════════════════════════════════════
//  BIQUAD FILTER DESIGN
// ═════════════════════════════════════════════

BiquadCoeff designHighPass(double fc, double fs) {
  BiquadCoeff c;
  double w0    = 2.0 * M_PI * fc / fs;
  double cosW0 = cos(w0);
  double alpha = sin(w0) / sqrt(2.0);
  double a0    = 1.0 + alpha;
  c.b0 =  (1.0 + cosW0) / 2.0 / a0;
  c.b1 = -(1.0 + cosW0)       / a0;
  c.b2 =  (1.0 + cosW0) / 2.0 / a0;
  c.a1 = (-2.0 * cosW0)       / a0;
  c.a2 =  (1.0 - alpha)       / a0;
  return c;
}

BiquadCoeff designLowPass(double fc, double fs) {
  BiquadCoeff c;
  double w0    = 2.0 * M_PI * fc / fs;
  double cosW0 = cos(w0);
  double alpha = sin(w0) / sqrt(2.0);
  double a0    = 1.0 + alpha;
  c.b0 = (1.0 - cosW0) / 2.0 / a0;
  c.b1 = (1.0 - cosW0)       / a0;
  c.b2 = (1.0 - cosW0) / 2.0 / a0;
  c.a1 = (-2.0 * cosW0)      / a0;
  c.a2 =  (1.0 - alpha)      / a0;
  return c;
}

inline float processBiquad(float x, const BiquadCoeff &c, BiquadState &s) {
  double y = c.b0 * x + c.b1 * s.x1 + c.b2 * s.x2
                      - c.a1 * s.y1  - c.a2 * s.y2;
  s.x2 = s.x1;  s.x1 = x;
  s.y2 = s.y1;  s.y1 = y;
  return (float)y;
}

// ═════════════════════════════════════════════
//  ADC INIT
// ═════════════════════════════════════════════

void initADCCalibration() {
  adc_cali_line_fitting_config_t cali_cfg = {
    .unit_id  = ADC_UNIT_NUM,
    .atten    = ADC_ATTEN_DB_12,
    .bitwidth = ADC_BITWIDTH_12,
  };
  esp_err_t err = adc_cali_create_scheme_line_fitting(&cali_cfg,
                                                      &adc_cali_handle);
  if (err == ESP_OK)
    Serial.println("[ADC] Line-fitting calibration active.");
  else {
    Serial.println("[ADC] Warning: calibration unavailable, using linear fallback.");
    adc_cali_handle = NULL;
  }
}

void initADC() {
  adc_continuous_handle_cfg_t handle_cfg = {
    .max_store_buf_size = CONV_FRAME_SIZE * 4,
    .conv_frame_size    = CONV_FRAME_SIZE,
  };
  ESP_ERROR_CHECK(adc_continuous_new_handle(&handle_cfg, &adc_cont_handle));

  adc_digi_pattern_config_t pattern = {
    .atten     = ADC_ATTEN_DB_12,
    .channel   = ADC_CHANNEL_NUM,
    .unit      = ADC_UNIT_NUM,
    .bit_width = SOC_ADC_DIGI_MAX_BITWIDTH,
  };

  adc_continuous_config_t cont_cfg = {
    .pattern_num    = 1,
    .adc_pattern    = &pattern,
    .sample_freq_hz = SAMPLE_RATE,
    .conv_mode      = ADC_CONV_SINGLE_UNIT_1,
    .format         = ADC_DIGI_OUTPUT_FORMAT_TYPE2,
  };
  ESP_ERROR_CHECK(adc_continuous_config(adc_cont_handle, &cont_cfg));
  ESP_ERROR_CHECK(adc_continuous_start(adc_cont_handle));
  Serial.println("[ADC] Continuous mode initialised.");
}

// ═════════════════════════════════════════════
//  FILTER PRE-WARM  (prevents start pop)
// ═════════════════════════════════════════════

void prewarmFilters(uint32_t durationMs) {
  uint32_t samplesToDiscard = (SAMPLE_RATE * durationMs) / 1000;
  uint8_t  raw[SOC_ADC_DIGI_RESULT_BYTES * 64];
  uint32_t discarded = 0;

  while (discarded < samplesToDiscard) {
    uint32_t bytesRead = 0;
    esp_err_t err = adc_continuous_read(adc_cont_handle,
                                        raw, sizeof(raw),
                                        &bytesRead,
                                        pdMS_TO_TICKS(500));
    if (err == ESP_ERR_TIMEOUT) {
      Serial.println("[ADC] Prewarm timeout — ADC not producing data!");
      return;
    }
    if (err != ESP_OK) continue;

    int results = bytesRead / SOC_ADC_DIGI_RESULT_BYTES;
    for (int i = 0; i < results && discarded < samplesToDiscard; i++) {
      adc_digi_output_data_t *p = (adc_digi_output_data_t*)
                                  &raw[i * SOC_ADC_DIGI_RESULT_BYTES];
      uint16_t rawVal = p->type2.data;
      int mv = 0;
      if (adc_cali_handle != NULL)
        adc_cali_raw_to_voltage(adc_cali_handle, rawVal, &mv);
      else
        mv = (int)((rawVal / 4095.0f) * 3300.0f);

      float s = ((float)mv - BIAS_MV) / BIAS_MV;
      if (g_hpEnabled)
        for (int st = 0; st < HP_STAGES; st++)
          s = processBiquad(s, g_hpCoeff, g_hpState[st]);
      if (g_lpEnabled)
        for (int st = 0; st < LP_STAGES; st++)
          s = processBiquad(s, g_lpCoeff, g_lpState[st]);
      discarded++;
    }
  }
}

// ═════════════════════════════════════════════
//  FILE COUNTER  (replaces SD.exists() scan)
// ═════════════════════════════════════════════

/**
 * Read the last used file number from COUNTER_FILE on the SD card.
 * Call once after the SD card is confirmed mounted.
 */
void loadFileCounter() {
  File f = SD.open(COUNTER_FILE, "r");
  if (!f) {
    g_lastFileNum = 0;
    Serial.println("[SD] No counter file found, starting from 1.");
    return;
  }
  g_lastFileNum = f.parseInt();
  f.close();
  Serial.printf("[SD] Loaded file counter: %d\n", g_lastFileNum);
}

/**
 * Write the current file number back to COUNTER_FILE.
 * Called at the start of each recording so even a power-loss mid-recording
 * leaves the counter correctly advanced.
 */
void saveFileCounter() {
  File f = SD.open(COUNTER_FILE, "w");
  if (!f) {
    Serial.println("[SD] Warning: could not save file counter.");
    return;
  }
  f.print(g_lastFileNum);
  f.close();
}

/**
 * Advance the counter and return the next filename. Find a free filename /rec_0001.wav … /rec_9999.wav
 */
String nextFilename() {
  g_lastFileNum++;
  char buf[20];
  snprintf(buf, sizeof(buf), "%s%04d.wav", FILENAME_BASE, g_lastFileNum);
  while (SD.exists(buf)) {
    g_lastFileNum++;
    snprintf(buf, sizeof(buf), "%s%04d.wav", FILENAME_BASE, g_lastFileNum);
  }
  return String(buf);
}

// ═════════════════════════════════════════════
//  WAV HELPERS
// ═════════════════════════════════════════════

void writeWavHeader(File &f) {
  uint32_t sampleRate  = SAMPLE_RATE;
  uint16_t channels    = CHANNELS;
  uint16_t bitsPerSamp = BITS_PER_SAMPLE;
  uint32_t byteRate    = sampleRate * channels * BYTES_PER_SAMPLE;
  uint16_t blockAlign  = channels * BYTES_PER_SAMPLE;
  uint16_t audioFmt    = 1;
  uint32_t placeholder = 0;
  uint32_t fmtSize     = 16;

  f.write((const uint8_t*)"RIFF", 4);
  f.write((uint8_t*)&placeholder, 4);
  f.write((const uint8_t*)"WAVE", 4);
  f.write((const uint8_t*)"fmt ", 4);
  f.write((uint8_t*)&fmtSize,     4);
  f.write((uint8_t*)&audioFmt,    2);
  f.write((uint8_t*)&channels,    2);
  f.write((uint8_t*)&sampleRate,  4);
  f.write((uint8_t*)&byteRate,    4);
  f.write((uint8_t*)&blockAlign,  2);
  f.write((uint8_t*)&bitsPerSamp, 2);
  f.write((const uint8_t*)"data", 4);
  f.write((uint8_t*)&placeholder, 4);
}

void patchWavHeader(File &f, uint32_t dataBytes) {
  uint32_t riffSize = dataBytes + WAV_HEADER_SIZE - 8;
  f.seek(4);   f.write((uint8_t*)&riffSize,  4);
  f.seek(40);  f.write((uint8_t*)&dataBytes, 4);
}

// ═════════════════════════════════════════════
//  RECORDING LOOP
// ═════════════════════════════════════════════

void recordAndFilter() {
  memset(g_hpState, 0, sizeof(g_hpState));
  memset(g_lpState, 0, sizeof(g_lpState));

  // ── Mount SD only if not already mounted ─────
  if (!g_sdMounted) {
    SD.end();
    digitalWrite(SD_CS_PIN, HIGH);
    delay(10);
    if (!SD.begin(SD_CS_PIN)) {
      Serial.println("[SD] Mount failed!");
      lcdShow("SD error!", "Check card");
      return;
    }
    g_sdMounted = true;
    loadFileCounter();
    Serial.println("[SD] Mounted OK.");
  } else {
    Serial.println("[SD] Using existing mount.");
  }

  // ── Advance counter and open file ────────────
  // Counter is saved BEFORE opening the file so a mid-recording
  // power loss still leaves the counter correctly advanced.
  String filename = nextFilename();
  saveFileCounter();

  Serial.printf("[SD] Opening %s\n", filename.c_str());
  g_wavFile = SD.open(filename, "w");
  if (!g_wavFile) {
    Serial.println("[SD] Open failed!");
    lcdShow("File error!", "Check card");
    g_sdMounted   = false;   // force remount next attempt
    g_lastFileNum--;         // roll back counter so the number isn't skipped
    return;
  }
  Serial.println("[SD] File opened OK.");

  // ── LCD: show filename ────────────────────────
  String displayName = filename.startsWith("/") ? filename.substring(1)
                                                : filename;
  char lcdFilename[LCD_COLS + 1];
  snprintf(lcdFilename, sizeof(lcdFilename), "%s", displayName.c_str());
  lcdShow(lcdFilename, "Prewarming...");

  // ── Write WAV header ─────────────────────────
  writeWavHeader(g_wavFile);
  g_dataBytesWritten = 0;

  // ── Pre-warm filters — only when HP filter is active ─
  // With FILTER_LOW_HZ = 0 the high-pass is disabled and there is no
  // filter settling transient, so the prewarm can be skipped entirely.
  if (g_hpEnabled) {
    uint32_t prewarmMs = max(200UL, 5000UL / FILTER_LOW_HZ);
    Serial.printf("[DSP] Pre-warming filters for %u ms...\n", prewarmMs);
    lcdShow(lcdFilename, "Prewarming...");
    prewarmFilters(prewarmMs);
  }

  // ── Allocate buffers ──────────────────────────
  int16_t *pcmBuf = (int16_t*)malloc(READ_BUF_SAMPLES * sizeof(int16_t));
  uint8_t *adcRaw = (uint8_t*)malloc(CONV_FRAME_SIZE);
  if (!pcmBuf || !adcRaw) {
    Serial.println("[MEM] Buffer allocation failed!");
    lcdShow("Memory error!", "");
    g_wavFile.close();
    free(pcmBuf);
    free(adcRaw);
    return;
  }

  uint32_t totalSamples   = (uint32_t)RECORD_SECONDS * SAMPLE_RATE;
  uint32_t samplesWritten = 0;
  bool     infinite       = (RECORD_SECONDS == 0);

  Serial.println("[REC] Starting... Press BOOT button to stop.");

  lcdLine2("Rec 0s");
  uint32_t lastLcdUpdate = millis();
  uint32_t recStartMs    = millis();

  while (infinite || samplesWritten < totalSamples) {

    // Stop button (active LOW)
    if (digitalRead(STOP_BUTTON_PIN) == LOW) {
      delay(50);
      if (digitalRead(STOP_BUTTON_PIN) == LOW) break;
    }

    // ── Read ADC ──────────────────────────────
    uint32_t bytesRead = 0;
    esp_err_t err = adc_continuous_read(adc_cont_handle,
                                        adcRaw,
                                        CONV_FRAME_SIZE,
                                        &bytesRead,
                                        pdMS_TO_TICKS(500));

    if (err == ESP_ERR_TIMEOUT) {
      Serial.println("[ADC] Read timeout — ADC not producing data!");
      lcdShow("ADC timeout!", "Check wiring");
      break;
    }
    if (err != ESP_OK) continue;

    int samplesRead = bytesRead / SOC_ADC_DIGI_RESULT_BYTES;

    // ── Process samples ───────────────────────
    for (int i = 0; i < samplesRead; i++) {
      adc_digi_output_data_t *p = (adc_digi_output_data_t*)
                                  &adcRaw[i * SOC_ADC_DIGI_RESULT_BYTES];
      uint16_t raw = p->type2.data;

      int mv = 0;
      if (adc_cali_handle != NULL)
        adc_cali_raw_to_voltage(adc_cali_handle, raw, &mv);
      else
        mv = (int)((raw / 4095.0f) * 3300.0f);

      float sample = ((float)mv - BIAS_MV) / BIAS_MV;

      // Filter first, then gain
      if (g_hpEnabled)
        for (int s = 0; s < HP_STAGES; s++)
          sample = processBiquad(sample, g_hpCoeff, g_hpState[s]);
      if (g_lpEnabled)
        for (int s = 0; s < LP_STAGES; s++)
          sample = processBiquad(sample, g_lpCoeff, g_lpState[s]);

      sample *= SOFTWARE_GAIN;
      sample  = constrain(sample, -1.0f, 1.0f);
      pcmBuf[i] = (int16_t)(sample * 32767.0f);
    }

    // ── Write to SD ───────────────────────────
    size_t bytesToWrite = samplesRead * BYTES_PER_SAMPLE;
    if (!infinite) {
      uint32_t remaining = totalSamples - samplesWritten;
      if ((uint32_t)samplesRead > remaining) {
        samplesRead  = remaining;
        bytesToWrite = samplesRead * BYTES_PER_SAMPLE;
      }
    }
    g_wavFile.write((uint8_t*)pcmBuf, bytesToWrite);
    g_dataBytesWritten += bytesToWrite;
    samplesWritten     += samplesRead;

    // ── Update LCD every second ───────────────
    uint32_t now = millis();
    if (now - lastLcdUpdate >= 1000) {
      lastLcdUpdate = now;

      uint32_t elapsedSec = (now - recStartMs) / 1000;
      char line2[LCD_COLS + 1];

      if (RECORD_SECONDS > 0) {
        snprintf(line2, sizeof(line2), "Rec %lus/%ds",
                 (unsigned long)elapsedSec, RECORD_SECONDS);
      } else {
        snprintf(line2, sizeof(line2), "Rec %lus [stop]",
                 (unsigned long)elapsedSec);
      }
      lcdLine2(line2);
      Serial.print('.');
    }
  }

  // ── Finalise WAV header ───────────────────────
  patchWavHeader(g_wavFile, g_dataBytesWritten);
  g_wavFile.close();
  free(pcmBuf);
  free(adcRaw);

  float dur = (float)g_dataBytesWritten /
              (float)(SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE);

  Serial.printf("\n[SD] Saved %s  (%.1f s, %u bytes)\n",
                filename.c_str(), dur, g_dataBytesWritten);

  char savedLine[LCD_COLS + 1];
  snprintf(savedLine, sizeof(savedLine), "Saved %.1fs", dur);
  lcdShow(lcdFilename, savedLine);
}

// ═════════════════════════════════════════════
//  SETUP & LOOP
// ═════════════════════════════════════════════

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== ESP32 Audio Filter Recorder ===");

  // ── I2C + LCD ────────────────────────────────
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  lcd.init();
  lcd.backlight();
  lcdShow("ESP32 Recorder", "Starting...");
  Serial.println("[LCD] Initialised.");
  delay(500);

  // ── Stop button ──────────────────────────────
  pinMode(STOP_BUTTON_PIN, INPUT_PULLUP);

  // ── Design filters ───────────────────────────
  g_hpEnabled = (FILTER_LOW_HZ  > 0 && FILTER_LOW_HZ  < SAMPLE_RATE / 2);
  g_lpEnabled = (FILTER_HIGH_HZ > 0 && FILTER_HIGH_HZ < SAMPLE_RATE / 2);

  if (g_hpEnabled) {
    g_hpCoeff = designHighPass(FILTER_LOW_HZ, SAMPLE_RATE);
    Serial.printf("[DSP] High-pass @ %d Hz enabled\n", FILTER_LOW_HZ);
  }
  if (g_lpEnabled) {
    g_lpCoeff = designLowPass(FILTER_HIGH_HZ, SAMPLE_RATE);
    Serial.printf("[DSP] Low-pass  @ %d Hz enabled\n", FILTER_HIGH_HZ);
  }
  if (!g_hpEnabled && !g_lpEnabled)
    Serial.println("[DSP] No filter active — recording raw audio");

  memset(g_hpState, 0, sizeof(g_hpState));
  memset(g_lpState, 0, sizeof(g_lpState));

  // ── SD card — first-time presence check ──────
  pinMode(SD_CS_PIN, OUTPUT);
  digitalWrite(SD_CS_PIN, HIGH);
  delay(10);
  SPI.begin();
  delay(250);
  lcdShow("ESP32 Recorder", "Checking SD...");
  if (!SD.begin(SD_CS_PIN)) {
    Serial.println("[SD] Initial check failed — will retry on first recording.");
    lcdShow("SD not found", "Will retry...");
    delay(1000);
  } else {
    Serial.printf("[SD] Card detected. Free: %llu MB\n",
                  (SD.totalBytes() - SD.usedBytes()) / (1024 * 1024));
    g_sdMounted = true;
    loadFileCounter();
  }

  // ── ADC init + calibration ───────────────────
  lcdShow("ESP32 Recorder", "Init ADC...");
  initADC();
  initADCCalibration();
  lcdShow("ESP32 Recorder", "Ready!");
  Serial.println("[SYS] Ready. Starting recording...\n");
}

void loop() {
  recordAndFilter();

  if (RECORD_SECONDS > 0) {
    Serial.println("[SYS] Recording complete. Waiting 2 s before next recording.");
    delay(2000);
    lcdShow("ESP32 Recorder", "Starting...");
    delay(500);
  } else {
    lcdShow("Press BOOT to", "record again");
    Serial.println("[SYS] Press BOOT to record again...");
    while (digitalRead(STOP_BUTTON_PIN) == HIGH) delay(50);
    delay(500);
    while (digitalRead(STOP_BUTTON_PIN) == LOW)  delay(50);
    delay(300);
    Serial.println("[SYS] Starting new recording...");
  }
}
