# InDEXt Loader

Offline LLM inference utility for legacy Android devices (**Android 2.3 – 4.4**, API 9–19), built without Android Studio using a pure C99 + ARM NEON runtime and native Android SDK UI.

Developed by **InDEX Integrated** · Version **2026.0.0**

---

## Features

- **Dual Format Support**
  - **IDXT**: Compact custom `Q8_0` format (28-byte header + block-32 `fp16` scale / `int8` weights) paired with HuggingFace `tokenizer.json`.
  - **GGUF (v2 / v3)**: Self-contained models with embedded vocabulary (`GPT-2 BPE` and `SentencePiece`). Supported quantizations: `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`.
- **Optimized for Low-RAM Legacy Hardware**
  - Zero-copy `mmap` weight loading directly from storage.
  - Persistent `pthread` worker pool and integer `Q8_0` quantized dot-products with **ARM NEON** (`armeabi-v7a`) plus clean scalar fallback (`armeabi`).
  - Automatic KV cache safety cap in `Model Default` mode to prevent 32-bit OOM on models with 32k/128k metadata context.
- **Native Adaptive UI**
  - Classic Android theme on Android 2.3; **Holo Dark / Holo Light** on Android 3.0–4.4.
  - Separate **Model Settings** (threads, context length, temperature, top-p, top-k, repeat penalty, seed, prompt mode, system prompt) and **App Settings** (theme, font size, auto-scroll, `tok/s` counter, timestamps, keep screen on, memory warning).
  - Supports **Completion** (raw continuation) and **Chat** (auto-detects `ChatML` `<|im_start|>`/`<|im_end|>` or `<|user|>`/`<|assistant|>`).

---

## Repository Structure

```text
InDEXtLoader/
├── AndroidManifest.xml       # minSdk 9 (Android 2.3), targetSdk 19 (Android 4.4)
├── build.bat                 # Windows one-click build wrapper
├── build_apk.py              # CLI build pipeline (NDK + aapt + javac + dx + zipalign + apksigner)
├── jni/
│   ├── Android.mk            # armeabi (scalar) + armeabi-v7a (ARM + NEON)
│   ├── Application.mk
│   ├── indext_engine.h / .c  # Transformer runtime, mmap loader, NEON kernels, sampler
│   ├── gguf_parser.h / .c    # GGUF v2/v3 parser & metadata reader
│   ├── tokenizer.h / .c      # BPE (GPT-2 ByteLevel & SentencePiece) tokenizer
│   ├── indext_jni.c          # JNI bridge (UTF-8 byte[] streaming)
│   └── cli_test.c            # Desktop CLI test runner
├── res/                      # App icons (mdpi/hdpi/xhdpi) and API-adaptive themes
└── src/com/indexintegrated/indextloader/
    ├── MainActivity.java
    ├── ChatActivity.java
    ├── ModelSettingsActivity.java
    ├── AppSettingsActivity.java
    ├── FilePickerActivity.java
    ├── GuideActivity.java
    ├── AboutActivity.java
    ├── NativeEngine.java
    ├── AppState.java
    └── UiHelper.java
```

---

## Building from Source (No Android Studio)

### Requirements
- **JDK 8+** (`javac`, `java`, `keytool`)
- **Python 3.8+**
- **Android NDK r16b** at `C:\Android\ndk\android-ndk-r16b` (for `armeabi` + `armeabi-v7a` / API 9–19 compatibility)
- **Android SDK** at `C:\Android\sdk` with `platforms/android-19` and `build-tools/28.0.3`

### Build APK
Run `build.bat` or:
```bat
python build_apk.py
```
Output: `InDEXtLoader-2026.0.0.apk` (signed with v1 JAR signature for Android 2.3–4.4 compatibility).

### Build Desktop CLI Test (Optional)
```bat
gcc -std=c99 -O3 jni/cli_test.c jni/indext_engine.c jni/gguf_parser.c jni/tokenizer.c -lm -o cli_test.exe
cli_test.exe path\to\model.idxt path\to\tokenizer.json "Once upon a time" 1
cli_test.exe path\to\model.gguf none "Once upon a time" 1
```

---

## Usage on Device

1. Install `InDEXtLoader-2026.0.0.apk` on your Android 2.3–4.4 device.
2. Copy your `.gguf` model (or `.idxt` + `tokenizer.json`) to `/sdcard/Download/`.
3. Open **InDEXt Loader**, tap **Start**, select **Format** (`IDXT` or `GGUF`), pick your file(s), and tap **Launch**.
4. Adjust sampling or prompt mode (**Completion** for base models, **Chat** for instruct models) in **Model Settings** anytime without reloading the model.
