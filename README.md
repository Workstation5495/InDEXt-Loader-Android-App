# InDEXt Loader

**InDEXt Loader** is a lightweight, offline Android LLM runner powered by the **InDEX-Transformer.C** native C99 engine. Designed for maximum compatibility and speed across both legacy Android devices (Android 2.3 / 4.x+) and modern smartphones, it runs **InDEX Transformer** (`IDXT`) and **GGUF** models directly on-device with zero internet connection.

## Links
- **GitHub**: [InDEXt-Loader-Android-App](https://github.com/Workstation5495/InDEXt-Loader-Android-App/)
- **HuggingFace**: [ResonateIntegrated/InDEXt-25M](https://huggingface.co/ResonateIntegrated/InDEXt-25M)

## Engine Highlights (`InDEX-Transformer.C`)
- **Formats**: Native **IDXT** (`IDX2` / `IDXT` + external `tokenizer.json`) and **GGUF** (v2 / v3 with embedded vocabulary).
- **All GGML Quantizations (ARM NEON + scalar)**: `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`.
- **4 KV Cache Modes**: `FP32`, `FP16` (2x less RAM), `Q8_0` (4x less RAM), and `Q4_0` (8x less RAM) with NEON-accelerated attention.
- **Architectures**: InDEX Transformer, LLaMA 1/2/3, Mistral, SmolLM / SmolLM2, Qwen 2 / 2.5 / 3 (with QK-Norm), Gemma 1 / 2 / 3 (embedding scaling, GeGLU, post-norms, logit softcapping), Phi-3 / 3.5 (fused QKV & gate/up), GPT-2, and StarCoder2.
- **Performance**: ARM NEON SIMD dot-products with dynamic `Q8_0` activation quantization, persistent `pthread` worker pool, cached RoPE, fast `fast_expf`, and selectable `mmap` zero-copy or full RAM loading.
- **Chat & Export**: Built-in templates (`Completion`, `Auto Chat`, `ChatML`, `LLaMA 3`, `Gemma`, `Alpaca`, `Phi / Zephyr`, `Mistral`) and conversation export to `.txt` or `.json`.

## Building
```bash
python build_apk.py
```

## Converting IDXT Models
```bash
python tools/convert_to_idxt2.py --all
```
