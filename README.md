# InDEXt Loader

**InDEXt Loader** is an offline Android application for running **InDEX Transformer** and open-weight large language models directly on-device with no internet connection, powered by the **InDEX-Transformer.C** native C99 inference engine.

Built to run reliably across both legacy Android devices (Android 2.3 / 4.x+) and modern smartphones, it combines a minimal, dependency-free Java UI with a SIMD-accelerated native engine.

## Links
- **GitHub**: [Workstation5495/InDEXt-Loader-Android-App](https://github.com/Workstation5495/InDEXt-Loader-Android-App/)
- **HuggingFace**: [ResonateIntegrated/InDEXt-25M](https://huggingface.co/ResonateIntegrated/InDEXt-25M)

## Engine Capabilities (`InDEX-Transformer.C`)
- **Model Formats**:
  - **IDXT** (`IDX2` / `IDXT` binary weights + external HuggingFace `tokenizer.json`)
  - **GGUF** (v2 / v3 self-contained models with embedded vocabulary and metadata)
- **Quantization Support (ARM NEON + Scalar)**:
  - **Floating Point**: `F32`, `F16`, `BF16`
  - **Legacy Block Quants**: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`
  - **K-Quants**: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`
- **KV Cache Precision Modes**:
  - `FP32` (full precision), `FP16` (2x less RAM), `Q8_0` (4x less RAM), and `Q4_0` (8x less RAM) with NEON-accelerated attention and automatic context sizing on memory-constrained devices.
- **Supported Architectures**:
  - **InDEX Transformer**, **LLaMA 1/2/3**, **Mistral**, **SmolLM / SmolLM2**, **TinyLlama**
  - **Qwen 2 / 2.5 / 3** (including QK-Norm and NeoX RoPE)
  - **Gemma 1 / 2 / 3** (embedding scaling, GeGLU, post-norms, attention/final logit softcapping)
  - **Phi-3 / 3.5** (fused QKV and fused gate/up projections)
  - **GPT-2 / StarCoder2 / Phi-2** (LayerNorm with bias and GELU MLP)
- **Runtime Optimizations**:
  - ARM NEON SIMD dot-products with dynamic `Q8_0` activation quantization
  - Persistent `pthread` worker pool, cached RoPE frequencies, fast `fast_expf`, and unrolled `matmul` loops
  - Toggleable `mmap` (zero-copy from storage) or full RAM buffer loading

## App Features
- **Prompt & Chat Templates**: `Completion (Raw)`, `Auto Chat`, `ChatML`, `LLaMA 3`, `Gemma`, `Alpaca`, `Phi / Zephyr`, and `Mistral`.
- **Sampling Controls**: Live adjustment of `CPU Threads`, `Max New Tokens`, `Temperature`, `Top-P`, `Top-K`, `Repeat Penalty`, and `Seed`.
- **Chat Export**: Export conversations to `.txt` or `.json` directly to `Download` or any selected folder.
- **Diagnostics & UI**: Load-time diagnostics toast (`mmap`/`RAM` + KV cache mode/size), live `tok/s` speed readout, low-memory warnings, and Dark / Light themes.

## Project Structure
- `jni/` — `InDEX-Transformer.c`, `indext_engine.h`, `gguf_parser.c/.h`, `tokenizer.c/.h`, `indext_jni.c`, `Android.mk`, `Application.mk`
- `src/` — Android Java activities and JNI bridge (`com.indexintegrated.indextloader`)
- `res/` — App icons and themes
- `build_apk.py`, `build.bat` — Automated NDK + SDK APK build and signing scripts

## Building from Source
Requirements: Python 3, Android NDK, Android SDK (`android-19` platform + `build-tools`), and JDK (`javac`, `keytool`).

```bash
python build_apk.py
```
The signed APK will be generated in the project root.
