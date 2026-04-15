/*
 * ESP32 Quad Piezo Band-Pass Filter with SD Card Recorder + I2C LCD
 * ==================================================================
 * Captures audio from FOUR analog piezo inputs simultaneously,
 * applies independent band-pass filters to each channel,
 * saves the result as a 4-channel WAV file on an SD card,
 * and displays recording status on a 16x2 I2C LCD.
 *
 * Channel assignment:
 *   Ch 1 (FL)  -> ADC1 CH7 -> GPIO 35   Piezo 1
 *   Ch 2 (FR)  -> ADC1 CH6 -> GPIO 34   Piezo 2
 *   Ch 3 (BL)  -> ADC1 CH0 -> GPIO 36   Piezo 3
 *   Ch 4 (BR)  -> ADC1 CH3 -> GPIO 39   Piezo 4
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
 * Required libraries (install via Library Manager):
 *   - LiquidCrystal I2C  (by Frank de Brabander)
 *   - SD (built-in with ESP32 Arduino core)
 *
 * WAV format: WAVE_FORMAT_EXTENSIBLE, 4-channel, 16-bit PCM.
 * Channel order in each frame: [Ch1][Ch2][Ch3][Ch4]
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

// ADC channels — all must be on ADC1 (ADC2 conflicts with Wi-Fi)
#define ADC_CH_1   ADC_CHANNEL_7   // GPIO35 — Piezo 1
#define ADC_CH_2   ADC_CHANNEL_6   // GPIO34 — Piezo 2
#define ADC_CH_3   ADC_CHANNEL_0   // GPIO36 — Piezo 3
#define ADC_CH_4   ADC_CHANNEL_3   // GPIO39 — Piezo 4
#define ADC_UNIT_NUM  ADC_UNIT_1
#define NUM_PIEZO     4            // number of active channels

// Recording duration in seconds. Set 0 for button-controlled recording.
#define RECORD_SECONDS      10

// Per-channel sample rate in Hz (minimum 20000 / NUM_PIEZO = 5000,
// but keep at 20000+ for good audio quality per channel).
// The ADC driver runs at SAMPLE_RATE * NUM_PIEZO internally and
// distributes evenly, so each channel gets exactly SAMPLE_RATE samples/sec.
#define SAMPLE_RATE         20000

#define BITS_PER_SAMPLE     16
#define CHANNELS            NUM_PIEZO   // 4 channels → stereo WAV extensible

// Band-pass filter cutoffs. Set either to 0 to disable that stage.
// Both cutoffs apply equally to all four channels.
#define FILTER_LOW_HZ       0
#define FILTER_HIGH_HZ      1000

// Cascaded biquad stages per filter (each adds 12 dB/oct roll-off)
#define HP_STAGES           3
#define LP_STAGES           3

// Software gain applied AFTER filtering
#define SOFTWARE_GAIN       1.0f

// ADC bias midpoint in millivolts (1650 mV for a centred 3.3V divider)
#define BIAS_MV             1650.0f

// Output filename base (/rec_0001.wav, /rec_0002.wav …)
#define FILENAME_BASE       "/rec_"
#define COUNTER_FILE        "/recnum.txt"

// ── Pin assignments ───────────────────────────
#define SD_CS_PIN           5
#define STOP_BUTTON_PIN     0    // GPIO0 = BOOT button

// ── LCD configuration ─────────────────────────
#define LCD_I2C_ADDRESS     0x27
#define LCD_COLS            16
#define LCD_ROWS            2
#define I2C_SDA_PIN         25
#define I2C_SCL_PIN         26

// ─────────────────────────────────────────────
//  INTERNAL CONSTANTS
// ─────────────────────────────────────────────

#define BYTES_PER_SAMPLE    (BITS_PER_SAMPLE / 8)

// 4-channel WAV uses WAVE_FORMAT_EXTENSIBLE which has a 68-byte header
// instead of the standard 44-byte header.
//   12 (RIFF) + 8 (fmt tag+size) + 40 (fmt extensible data) + 8 (data tag+size)
#define WAV_HEADER_SIZE     68

// Offset of the data-chunk size field in the extensible header
#define WAV_DATA_SIZE_OFFSET  64

#define READ_BUF_SAMPLES    1024
// ADC runs at 4x the per-channel rate; frame buffer holds one sample
// per channel per ADC result burst.
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

static LiquidCrystal_I2C lcd(LCD_I2C_ADDRESS, LCD_COLS, LCD_ROWS);

static File     g_wavFile;
static uint32_t g_dataBytesWritten = 0;

// Shared filter coefficients (same cutoffs for all channels)
static BiquadCoeff g_hpCoeff, g_lpCoeff;
static bool        g_hpEnabled = false;
static bool        g_lpEnabled = false;

// Independent filter state per channel per stage: [channel][stage]
static BiquadState g_hpState[NUM_PIEZO][HP_STAGES];
static BiquadState g_lpState[NUM_PIEZO][LP_STAGES];

static adc_continuous_handle_t adc_cont_handle = NULL;
static adc_cali_handle_t       adc_cali_handle = NULL;

static bool g_sdMounted   = false;
static int  g_lastFileNum = 0;

// Maps ADC channel enum to output channel index 0–3 and vice versa
static const adc_channel_t k_adcChannels[NUM_PIEZO] = {
  ADC_CH_1, ADC_CH_2, ADC_CH_3, ADC_CH_4
};

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

// ─────────────────────────────────────────────
//  SAMPLE PROCESSING HELPER
// ─────────────────────────────────────────────

/**
 * Convert a raw 12-bit ADC value to a filtered, gained float in -1..+1.
 * ch: 0–3 selects independent filter state for each piezo channel.
 */
float processSample(uint16_t raw, int ch) {
  int mv = 0;
  if (adc_cali_handle != NULL)
    adc_cali_raw_to_voltage(adc_cali_handle, raw, &mv);
  else
    mv = (int)((raw / 4095.0f) * 3300.0f);

  float sample = ((float)mv - BIAS_MV) / BIAS_MV;

  // Filter first, then gain
  if (g_hpEnabled)
    for (int s = 0; s < HP_STAGES; s++)
      sample = processBiquad(sample, g_hpCoeff, g_hpState[ch][s]);
  if (g_lpEnabled)
    for (int s = 0; s < LP_STAGES; s++)
      sample = processBiquad(sample, g_lpCoeff, g_lpState[ch][s]);

  sample *= SOFTWARE_GAIN;
  return constrain(sample, -1.0f, 1.0f);
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

  // Configure all four channels. The driver interleaves their results
  // and divides sample_freq_hz evenly, so each channel gets SAMPLE_RATE
  // samples per second when sample_freq_hz = SAMPLE_RATE * NUM_PIEZO.
  adc_digi_pattern_config_t patterns[NUM_PIEZO];
  for (int i = 0; i < NUM_PIEZO; i++) {
    patterns[i].atten     = ADC_ATTEN_DB_12;
    patterns[i].channel   = k_adcChannels[i];
    patterns[i].unit      = ADC_UNIT_NUM;
    patterns[i].bit_width = SOC_ADC_DIGI_MAX_BITWIDTH;
  }

  adc_continuous_config_t cont_cfg = {
    .pattern_num    = NUM_PIEZO,
    .adc_pattern    = patterns,
    .sample_freq_hz = (uint32_t)SAMPLE_RATE * NUM_PIEZO,
    .conv_mode      = ADC_CONV_SINGLE_UNIT_1,
    .format         = ADC_DIGI_OUTPUT_FORMAT_TYPE2,
  };
  ESP_ERROR_CHECK(adc_continuous_config(adc_cont_handle, &cont_cfg));
  ESP_ERROR_CHECK(adc_continuous_start(adc_cont_handle));
  Serial.printf("[ADC] Continuous mode initialised (%d channels @ %d Hz each).\n",
                NUM_PIEZO, SAMPLE_RATE);
}

// ═════════════════════════════════════════════
//  FILTER PRE-WARM  (prevents start pop)
// ═════════════════════════════════════════════

void prewarmFilters(uint32_t durationMs) {
  uint32_t samplesToDiscard = ((uint32_t)SAMPLE_RATE * NUM_PIEZO * durationMs) / 1000;
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
      // Determine which channel this result belongs to
      int ch = -1;
      for (int c = 0; c < NUM_PIEZO; c++) {
        if (p->type2.channel == (uint8_t)k_adcChannels[c]) { ch = c; break; }
      }
      if (ch < 0) continue;
      processSample(p->type2.data, ch);   // run through filter, discard output
      discarded++;
    }
  }
}

// ═════════════════════════════════════════════
//  FILE COUNTER
// ═════════════════════════════════════════════

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

void saveFileCounter() {
  File f = SD.open(COUNTER_FILE, "w");
  if (!f) {
    Serial.println("[SD] Warning: could not save file counter.");
    return;
  }
  f.print(g_lastFileNum);
  f.close();
}

// O(1) — no SD.exists() scan. Counter is the source of truth.
String nextFilename() {
  g_lastFileNum++;
  char buf[20];
  snprintf(buf, sizeof(buf), "%s%04d.wav", FILENAME_BASE, g_lastFileNum);
  return String(buf);
}

// ═════════════════════════════════════════════
//  WAV HELPERS  (WAVE_FORMAT_EXTENSIBLE for 4 ch)
// ═════════════════════════════════════════════

/**
 * Write a 68-byte WAVE_FORMAT_EXTENSIBLE header.
 * Required for WAV files with more than 2 channels.
 * Data size fields are placeholders — patched after recording.
 */
void writeWavHeader(File &f) {
  uint32_t sampleRate  = SAMPLE_RATE;
  uint16_t channels    = CHANNELS;         // 4
  uint16_t bitsPerSamp = BITS_PER_SAMPLE;  // 16
  uint32_t byteRate    = sampleRate * channels * BYTES_PER_SAMPLE;
  uint16_t blockAlign  = channels * BYTES_PER_SAMPLE;
  uint32_t placeholder = 0;

  // WAVE_FORMAT_EXTENSIBLE tag and fmt chunk size (40 bytes of data)
  uint16_t fmtTag  = 0xFFFE;
  uint32_t fmtSize = 40;

  // Extension fields
  uint16_t cbSize      = 22;   // bytes after wBitsPerSample in fmt chunk
  uint16_t validBits   = 16;
  // Channel mask: FL | FR | BL | BR
  uint32_t chanMask    = 0x00000033;
  // SubFormat GUID for PCM: {00000001-0000-0010-8000-00AA00389B71}
  uint8_t  subFmt[16]  = {
    0x01,0x00,0x00,0x00, 0x00,0x00, 0x10,0x00,
    0x80,0x00, 0x00,0xAA,0x00,0x38,0x9B,0x71
  };

  // RIFF chunk (12 bytes)
  f.write((const uint8_t*)"RIFF", 4);
  f.write((uint8_t*)&placeholder, 4);   // RIFF size — patched later
  f.write((const uint8_t*)"WAVE", 4);

  // fmt chunk (8 + 40 = 48 bytes)
  f.write((const uint8_t*)"fmt ", 4);
  f.write((uint8_t*)&fmtSize,    4);
  f.write((uint8_t*)&fmtTag,     2);
  f.write((uint8_t*)&channels,   2);
  f.write((uint8_t*)&sampleRate, 4);
  f.write((uint8_t*)&byteRate,   4);
  f.write((uint8_t*)&blockAlign, 2);
  f.write((uint8_t*)&bitsPerSamp,2);
  f.write((uint8_t*)&cbSize,     2);
  f.write((uint8_t*)&validBits,  2);
  f.write((uint8_t*)&chanMask,   4);
  f.write(subFmt,               16);

  // data chunk header (8 bytes)
  f.write((const uint8_t*)"data", 4);
  f.write((uint8_t*)&placeholder, 4);   // data size — patched later
}

/**
 * Patch the RIFF and data size fields after recording completes.
 * Offsets are specific to the 68-byte extensible header.
 */
void patchWavHeader(File &f, uint32_t dataBytes) {
  uint32_t riffSize = dataBytes + WAV_HEADER_SIZE - 8;
  f.seek(4);
  f.write((uint8_t*)&riffSize,  4);
  f.seek(WAV_DATA_SIZE_OFFSET);
  f.write((uint8_t*)&dataBytes, 4);
}

// ═════════════════════════════════════════════
//  RECORDING LOOP
// ═════════════════════════════════════════════

void recordAndFilter() {
  memset(g_hpState, 0, sizeof(g_hpState));
  memset(g_lpState, 0, sizeof(g_lpState));

  // ── Mount SD only when needed ────────────────
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

  // ── Get next filename and persist counter ─────
  // Counter saved before opening the file — safe against mid-recording
  // power loss.
  String filename = nextFilename();
  saveFileCounter();

  Serial.printf("[SD] Opening %s\n", filename.c_str());
  g_wavFile = SD.open(filename, "w");
  if (!g_wavFile) {
    Serial.println("[SD] Open failed!");
    lcdShow("File error!", "Check card");
    g_sdMounted   = false;
    g_lastFileNum--;   // roll back so the number isn't skipped
    return;
  }
  Serial.println("[SD] File opened OK.");

  // ── LCD: filename on line 1 ───────────────────
  String displayName = filename.startsWith("/") ? filename.substring(1)
                                                : filename;
  char lcdFilename[LCD_COLS + 1];
  snprintf(lcdFilename, sizeof(lcdFilename), "%s", displayName.c_str());
  lcdShow(lcdFilename, "4ch recording");

  // ── Write WAV header ─────────────────────────
  writeWavHeader(g_wavFile);
  g_dataBytesWritten = 0;

  // ── Pre-warm filters (only if HP is active) ───
  if (g_hpEnabled) {
    uint32_t prewarmMs = max(200UL, 5000UL / FILTER_LOW_HZ);
    Serial.printf("[DSP] Pre-warming filters for %u ms...\n", prewarmMs);
    lcdShow(lcdFilename, "Prewarming...");
    prewarmFilters(prewarmMs);
  }

  // ── Allocate buffers ──────────────────────────
  // pcmBuf holds interleaved 4-channel frames:
  // [Ch1][Ch2][Ch3][Ch4][Ch1][Ch2][Ch3][Ch4]...
  // Worst case: READ_BUF_SAMPLES total ADC results / 4 channels =
  // READ_BUF_SAMPLES/4 complete frames × 4 samples each = READ_BUF_SAMPLES.
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

  // Total 4-channel frames to record
  uint32_t totalFrames   = (uint32_t)RECORD_SECONDS * SAMPLE_RATE;
  uint32_t framesWritten = 0;
  bool     infinite      = (RECORD_SECONDS == 0);

  Serial.println("[REC] Starting... Press BOOT button to stop.");
  lcdLine2("Rec 0s");

  uint32_t lastLcdUpdate = millis();
  uint32_t recStartMs    = millis();

  // Pending frame: holds the latest sample from each channel until
  // all four have arrived, then the complete frame is written.
  int16_t  pendingFrame[NUM_PIEZO] = {0};
  uint8_t  pendingMask             = 0;   // bit i set when channel i received

  while (infinite || framesWritten < totalFrames) {

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

    int totalResults = bytesRead / SOC_ADC_DIGI_RESULT_BYTES;
    int outIdx       = 0;

    // ── Route and process each ADC result ────
    for (int i = 0; i < totalResults; i++) {
      adc_digi_output_data_t *p = (adc_digi_output_data_t*)
                                  &adcRaw[i * SOC_ADC_DIGI_RESULT_BYTES];
      uint8_t chanId = p->type2.channel;

      // Map ADC channel ID to output channel index 0–3
      int ch = -1;
      for (int c = 0; c < NUM_PIEZO; c++) {
        if (chanId == (uint8_t)k_adcChannels[c]) { ch = c; break; }
      }
      if (ch < 0) continue;   // unknown channel — skip

      pendingFrame[ch]   = (int16_t)(processSample(p->type2.data, ch) * 32767.0f);
      pendingMask       |= (1 << ch);

      // When all four channels have contributed, write the complete frame
      if (pendingMask == 0x0F) {
        if (outIdx + NUM_PIEZO <= READ_BUF_SAMPLES) {
          for (int c = 0; c < NUM_PIEZO; c++)
            pcmBuf[outIdx++] = pendingFrame[c];
        }
        pendingMask = 0;   // reset for the next frame
      }
    }

    // ── Write complete frames to SD ───────────
    // outIdx is always a multiple of NUM_PIEZO here
    if (outIdx > 0) {
      uint32_t framesThisRead = outIdx / NUM_PIEZO;

      if (!infinite) {
        uint32_t remaining = totalFrames - framesWritten;
        if (framesThisRead > remaining) {
          framesThisRead = remaining;
          outIdx = framesThisRead * NUM_PIEZO;
        }
      }

      g_wavFile.write((uint8_t*)pcmBuf, outIdx * BYTES_PER_SAMPLE);
      g_dataBytesWritten += outIdx * BYTES_PER_SAMPLE;
      framesWritten      += framesThisRead;
    }

    // ── Update LCD every second ───────────────
    uint32_t now = millis();
    if (now - lastLcdUpdate >= 1000) {
      lastLcdUpdate = now;
      uint32_t elapsedSec = (now - recStartMs) / 1000;
      char line2[LCD_COLS + 1];
      if (RECORD_SECONDS > 0)
        snprintf(line2, sizeof(line2), "Rec %lus/%ds",
                 (unsigned long)elapsedSec, RECORD_SECONDS);
      else
        snprintf(line2, sizeof(line2), "Rec %lus [stop]",
                 (unsigned long)elapsedSec);
      lcdLine2(line2);
      Serial.print('.');
    }
  }

  // ── Finalise ─────────────────────────────────
  patchWavHeader(g_wavFile, g_dataBytesWritten);
  g_wavFile.close();
  free(pcmBuf);
  free(adcRaw);

  // Duration: total bytes / (frames/sec × bytes/frame)
  float dur = (float)g_dataBytesWritten /
              (float)(SAMPLE_RATE * NUM_PIEZO * BYTES_PER_SAMPLE);

  Serial.printf("\n[SD] Saved %s  (%.1f s, %u bytes, %d-channel)\n",
                filename.c_str(), dur, g_dataBytesWritten, NUM_PIEZO);

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
  Serial.println("\n=== ESP32 Quad Piezo Filter Recorder ===");

  // ── I2C + LCD ────────────────────────────────
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  lcd.init();
  lcd.backlight();
  lcdShow("ESP32 Recorder", "Starting...");
  Serial.println("[LCD] Initialised.");

  // Single 500 ms delay covers I2C/SPI peripheral settle and SD power-on
  delay(500);

  // ── Stop button ──────────────────────────────
  pinMode(STOP_BUTTON_PIN, INPUT_PULLUP);

  // ── Design filters ───────────────────────────
  g_hpEnabled = (FILTER_LOW_HZ  > 0 && FILTER_LOW_HZ  < SAMPLE_RATE / 2);
  g_lpEnabled = (FILTER_HIGH_HZ > 0 && FILTER_HIGH_HZ < SAMPLE_RATE / 2);

  if (g_hpEnabled) {
    g_hpCoeff = designHighPass(FILTER_LOW_HZ, SAMPLE_RATE);
    Serial.printf("[DSP] High-pass @ %d Hz  (%d stages, %d dB/oct)\n",
                  FILTER_LOW_HZ, HP_STAGES, HP_STAGES * 12);
  }
  if (g_lpEnabled) {
    g_lpCoeff = designLowPass(FILTER_HIGH_HZ, SAMPLE_RATE);
    Serial.printf("[DSP] Low-pass  @ %d Hz  (%d stages, %d dB/oct)\n",
                  FILTER_HIGH_HZ, LP_STAGES, LP_STAGES * 12);
  }
  if (!g_hpEnabled && !g_lpEnabled)
    Serial.println("[DSP] No filter active — recording raw audio");

  memset(g_hpState, 0, sizeof(g_hpState));
  memset(g_lpState, 0, sizeof(g_lpState));

  // ── SD card ──────────────────────────────────
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

  // ── ADC ──────────────────────────────────────
  lcdShow("ESP32 Recorder", "Init ADC...");
  initADC();
  initADCCalibration();

  lcdShow("ESP32 Recorder", "Ready! (4ch)");
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
