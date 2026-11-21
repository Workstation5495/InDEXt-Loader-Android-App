#include "indext_engine.h"
#include "gguf_parser.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <pthread.h>
#endif

/* ---- Timing & CPU count ---- */
static double now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
#endif
}

static int detect_cpu_cores(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 2;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 2;
    if (n > 8) n = 8;
    return (int)n;
#endif
}

/* ---- Fast expf (Cody-Waite + degree-5 minimax, ~8x faster than libm expf on ARMv7) ---- */
static inline float fast_expf(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    float kf = floorf(x * 1.4426950408889634f + 0.5f);
    int32_t k = (int32_t)kf;
    float r = (x - kf * 0.693145751953125f) - kf * 1.4286068203094172e-6f;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * 0.008333334f))));
    uint32_t pow2k = (uint32_t)(k + 127) << 23;
    float scale;
    memcpy(&scale, &pow2k, 4);
    return p * scale;
}

/* ---- FP16 <-> FP32 & BF16 ---- */
static float fp16_lut[65536];
static int fp16_lut_ready = 0;

static float fp16_bits_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FFu;
            f = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

static void init_fp16_lut(void) {
    if (fp16_lut_ready) return;
    for (uint32_t i = 0; i < 65536; i++) {
        fp16_lut[i] = fp16_bits_to_fp32((uint16_t)i);
    }
    fp16_lut_ready = 1;
}

static inline float fp16_to_fp32(uint16_t h) { return fp16_lut[h]; }

static inline uint16_t fp32_to_fp16(float x) {
    uint32_t f;
    memcpy(&f, &x, 4);
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = (int32_t)((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = (f >> 13) & 0x03FFu;
    if (exp <= 0) {
        return (uint16_t)sign;
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7BFFu);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static inline float bf16_to_fp32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float out;
    memcpy(&out, &u, 4);
    return out;
}

/* ---- Tensor representation & row size ---- */
typedef struct {
    const uint8_t *data;
    uint32_t type;
    uint32_t rows;
    uint32_t cols;
} TensorView;

static int is_supported_quant_type(uint32_t type) {
    switch (type) {
        case GGML_T_F32:
        case GGML_T_F16:
        case GGML_T_BF16:
        case GGML_T_Q4_0:
        case GGML_T_Q4_1:
        case GGML_T_Q5_0:
        case GGML_T_Q5_1:
        case GGML_T_Q8_0:
        case GGML_T_Q2_K:
        case GGML_T_Q3_K:
        case GGML_T_Q4_K:
        case GGML_T_Q5_K:
        case GGML_T_Q6_K:
            return 1;
        default:
            return 0;
    }
}

static size_t tensor_row_bytes(uint32_t type, uint32_t cols) {
    switch (type) {
        case GGML_T_F32:  return (size_t)cols * 4;
        case GGML_T_F16:  return (size_t)cols * 2;
        case GGML_T_BF16: return (size_t)cols * 2;
        case GGML_T_Q4_0: return (size_t)(cols / 32) * 18;
        case GGML_T_Q4_1: return (size_t)(cols / 32) * 20;
        case GGML_T_Q5_0: return (size_t)(cols / 32) * 22;
        case GGML_T_Q5_1: return (size_t)(cols / 32) * 24;
        case GGML_T_Q8_0: return (size_t)(cols / 32) * 34;
        case GGML_T_Q2_K: return (size_t)(cols / 256) * 84;
        case GGML_T_Q3_K: return (size_t)(cols / 256) * 110;
        case GGML_T_Q4_K: return (size_t)(cols / 256) * 144;
        case GGML_T_Q5_K: return (size_t)(cols / 256) * 176;
        case GGML_T_Q6_K: return (size_t)(cols / 256) * 210;
        default: return 0;
    }
}

/* ---- Row dequantization (embeddings / 1D norms) ---- */
static void dequant_row(const TensorView *t, uint32_t row, float *out) {
    uint32_t n = t->cols;
    size_t rb = tensor_row_bytes(t->type, n);
    const uint8_t *p = t->data + (size_t)row * rb;

    if (t->type == GGML_T_F32) {
        memcpy(out, p, (size_t)n * 4);
    } else if (t->type == GGML_T_F16) {
        const uint16_t *h = (const uint16_t *)p;
        for (uint32_t i = 0; i < n; i++) out[i] = fp16_to_fp32(h[i]);
    } else if (t->type == GGML_T_BF16) {
        const uint16_t *b = (const uint16_t *)p;
        for (uint32_t i = 0; i < n; i++) out[i] = bf16_to_fp32(b[i]);
    } else if (t->type == GGML_T_Q8_0) {
        uint32_t nb = n / 32;
        for (uint32_t b = 0; b < nb; b++) {
            uint16_t d16; memcpy(&d16, p, 2);
            float d = fp16_to_fp32(d16);
            const int8_t *qs = (const int8_t *)(p + 2);
            for (int j = 0; j < 32; j++) out[b * 32 + j] = d * (float)qs[j];
            p += 34;
        }
    } else if (t->type == GGML_T_Q4_0) {
        uint32_t nb = n / 32;
        for (uint32_t b = 0; b < nb; b++) {
            uint16_t d16; memcpy(&d16, p, 2);
            float d = fp16_to_fp32(d16);
            const uint8_t *qs = p + 2;
            for (int j = 0; j < 16; j++) {
                out[b * 32 + j]      = d * (float)((int)(qs[j] & 0x0F) - 8);
                out[b * 32 + j + 16] = d * (float)((int)(qs[j] >> 4)   - 8);
            }
            p += 18;
        }
    } else if (t->type == GGML_T_Q4_1) {
        uint32_t nb = n / 32;
        for (uint32_t b = 0; b < nb; b++) {
            uint16_t d16, m16; memcpy(&d16, p, 2); memcpy(&m16, p + 2, 2);
            float d = fp16_to_fp32(d16), m = fp16_to_fp32(m16);
            const uint8_t *qs = p + 4;
            for (int j = 0; j < 16; j++) {
                out[b * 32 + j]      = d * (float)(qs[j] & 0x0F) + m;
                out[b * 32 + j + 16] = d * (float)(qs[j] >> 4) + m;
            }
            p += 20;
        }
    } else if (t->type == GGML_T_Q5_0) {
        uint32_t nb = n / 32;
        for (uint32_t b = 0; b < nb; b++) {
            uint16_t d16; uint32_t qh;
            memcpy(&d16, p, 2); memcpy(&qh, p + 2, 4);
            float d = fp16_to_fp32(d16);
            const uint8_t *qs = p + 6;
            for (int j = 0; j < 16; j++) {
                uint8_t xh_0 = ((qh >> j) << 4) & 0x10;
                uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
                int x0 = ((qs[j] & 0x0F) | xh_0) - 16;
                int x1 = ((qs[j] >> 4)   | xh_1) - 16;
                out[b * 32 + j]      = d * (float)x0;
                out[b * 32 + j + 16] = d * (float)x1;
            }
            p += 22;
        }
    } else if (t->type == GGML_T_Q5_1) {
        uint32_t nb = n / 32;
        for (uint32_t b = 0; b < nb; b++) {
            uint16_t d16, m16; uint32_t qh;
            memcpy(&d16, p, 2); memcpy(&m16, p + 2, 2); memcpy(&qh, p + 4, 4);
            float d = fp16_to_fp32(d16), m = fp16_to_fp32(m16);
            const uint8_t *qs = p + 8;
            for (int j = 0; j < 16; j++) {
                uint8_t xh_0 = ((qh >> j) << 4) & 0x10;
                uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
                out[b * 32 + j]      = d * (float)((qs[j] & 0x0F) | xh_0) + m;
                out[b * 32 + j + 16] = d * (float)((qs[j] >> 4)   | xh_1) + m;
            }
            p += 24;
        }
    } else if (t->type == GGML_T_Q2_K) {
        uint32_t nb = n / 256;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t *scales = p;
            const uint8_t *qs = p + 16;
            uint16_t d16, dmin16;
            memcpy(&d16, p + 80, 2); memcpy(&dmin16, p + 82, 2);
            float d = fp16_to_fp32(d16), min = fp16_to_fp32(dmin16);
            float *ob = out + b * 256;
            int is = 0;
            for (int n128 = 0; n128 < 256; n128 += 128) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    uint8_t sc = scales[is++];
                    float dl = d * (float)(sc & 0xF), ml = min * (float)(sc >> 4);
                    for (int l = 0; l < 16; l++) ob[n128 + j * 32 + l] = dl * (float)((qs[l] >> shift) & 3) - ml;
                    sc = scales[is++];
                    dl = d * (float)(sc & 0xF); ml = min * (float)(sc >> 4);
                    for (int l = 0; l < 16; l++) ob[n128 + j * 32 + 16 + l] = dl * (float)((qs[l + 16] >> shift) & 3) - ml;
                    shift += 2;
                }
                qs += 32;
            }
            p += 84;
        }
    } else if (t->type == GGML_T_Q3_K) {
        uint32_t nb = n / 256;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t *hmask = p;
            const uint8_t *qs = p + 32;
            uint32_t aux[4];
            memcpy(aux, p + 96, 12);
            uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 4) & 0x0F0F0F0Fu) << 4);
            aux[3] = ((aux[1] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 0) & 0x0F0F0F0Fu) << 4);
            aux[0] = (aux[0] & 0x0F0F0F0Fu) | (((tmp >> 0) & 0x03030303u) << 4);
            aux[1] = (aux[1] & 0x0F0F0F0Fu) | (((tmp >> 2) & 0x03030303u) << 4);
            const int8_t *scales = (const int8_t *)aux;
            uint16_t d16; memcpy(&d16, p + 108, 2);
            float d_all = fp16_to_fp32(d16);
            float *ob = out + b * 256;
            int is = 0; uint8_t m = 1;
            for (int n128 = 0; n128 < 256; n128 += 128) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    float dl = d_all * (float)(scales[is++] - 32);
                    for (int l = 0; l < 16; l++) {
                        int q = (int)((qs[l] >> shift) & 3) - ((hmask[l] & m) ? 0 : 4);
                        ob[n128 + j * 32 + l] = dl * (float)q;
                    }
                    dl = d_all * (float)(scales[is++] - 32);
                    for (int l = 0; l < 16; l++) {
                        int q = (int)((qs[l + 16] >> shift) & 3) - ((hmask[l + 16] & m) ? 0 : 4);
                        ob[n128 + j * 32 + 16 + l] = dl * (float)q;
                    }
                    shift += 2; m <<= 1;
                }
                qs += 32;
            }
            p += 110;
        }
    } else if (t->type == GGML_T_Q4_K) {
        uint32_t nb = n / 256;
        for (uint32_t b = 0; b < nb; b++) {
            uint16_t d16, dmin16;
            memcpy(&d16, p, 2); memcpy(&dmin16, p + 2, 2);
            float d = fp16_to_fp32(d16), dmin = fp16_to_fp32(dmin16);
            const uint8_t *scales = p + 4;
            const uint8_t *qs = p + 16;
            float *ob = out + b * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                if (is < 4) { sc1 = scales[is] & 63; m1 = scales[is + 4] & 63; }
                else        { sc1 = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4); m1 = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4); }
                if (is + 1 < 4) { sc2 = scales[is + 1] & 63; m2 = scales[is + 5] & 63; }
                else            { sc2 = (scales[is + 5] & 0xF) | ((scales[is - 3] >> 6) << 4); m2 = (scales[is + 5] >> 4) | ((scales[is + 1] >> 6) << 4); }
                float d1 = d * sc1, min1 = dmin * m1;
                float d2 = d * sc2, min2 = dmin * m2;
                for (int l = 0; l < 32; l++) ob[j + l]      = d1 * (float)(qs[l] & 0xF) - min1;
                for (int l = 0; l < 32; l++) ob[j + 32 + l] = d2 * (float)(qs[l] >> 4)  - min2;
                qs += 32;
                is += 2;
            }
            p += 144;
        }
    } else if (t->type == GGML_T_Q5_K) {
        uint32_t nb = n / 256;
        for (uint32_t b = 0; b < nb; b++) {
            uint16_t d16, dmin16;
            memcpy(&d16, p, 2); memcpy(&dmin16, p + 2, 2);
            float d = fp16_to_fp32(d16), dmin = fp16_to_fp32(dmin16);
            const uint8_t *scales = p + 4;
            const uint8_t *qh = p + 16;
            const uint8_t *qs = p + 48;
            float *ob = out + b * 256;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc1, m1, sc2, m2;
                if (is < 4) { sc1 = scales[is] & 63; m1 = scales[is + 4] & 63; }
                else        { sc1 = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4); m1 = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4); }
                if (is + 1 < 4) { sc2 = scales[is + 1] & 63; m2 = scales[is + 5] & 63; }
                else            { sc2 = (scales[is + 5] & 0xF) | ((scales[is - 3] >> 6) << 4); m2 = (scales[is + 5] >> 4) | ((scales[is + 1] >> 6) << 4); }
                float d1 = d * sc1, min1 = dmin * m1;
                float d2 = d * sc2, min2 = dmin * m2;
                for (int l = 0; l < 32; l++) {
                    ob[j + l]      = d1 * (float)((qs[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - min1;
                    ob[j + 32 + l] = d2 * (float)((qs[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - min2;
                }
                qs += 32; is += 2; u1 <<= 2; u2 <<= 2;
            }
            p += 176;
        }
    } else if (t->type == GGML_T_Q6_K) {
        uint32_t nb = n / 256;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t *ql = p;
            const uint8_t *qh = p + 128;
            const int8_t  *sc = (const int8_t *)(p + 192);
            uint16_t d16; memcpy(&d16, p + 208, 2);
            float d = fp16_to_fp32(d16);
            float *ob = out + b * 256;
            for (int n128 = 0; n128 < 256; n128 += 128) {
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int8_t q1 = (int8_t)((ql[l + 0]  & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l + 0]  >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                    ob[n128 + l + 0]  = d * sc[is + 0] * q1;
                    ob[n128 + l + 32] = d * sc[is + 2] * q2;
                    ob[n128 + l + 64] = d * sc[is + 4] * q3;
                    ob[n128 + l + 96] = d * sc[is + 6] * q4;
                }
                ql += 64; qh += 32; sc += 8;
            }
            p += 210;
        }
    }
}

/* ---- Dynamic Q8_0 activation quantization for fast integer dot products ---- */
typedef struct {
    float d;
    float sumf;
    int8_t qs[32];
} BlockQ8;

static int needs_q8_x(uint32_t type) {
    return (type != GGML_T_F32 && type != GGML_T_F16 && type != GGML_T_BF16);
}

static void quantize_x_q8(const float *x, BlockQ8 *qx, uint32_t n) {
    uint32_t nb = n / 32;
    for (uint32_t b = 0; b < nb; b++) {
        const float *xb = x + b * 32;
        float amax = 0.0f;
        float sumf = 0.0f;
#ifdef __ARM_NEON
        float32x4_t vmax = vdupq_n_f32(0.0f);
        float32x4_t vsum = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 4) {
            float32x4_t vx = vld1q_f32(xb + j);
            vmax = vmaxq_f32(vmax, vabsq_f32(vx));
            vsum = vaddq_f32(vsum, vx);
        }
        float32x2_t m2 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        m2 = vpmax_f32(m2, m2);
        amax = vget_lane_f32(m2, 0);
        float32x2_t s2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
        sumf = vget_lane_f32(vpadd_f32(s2, s2), 0);
#else
        for (int j = 0; j < 32; j++) {
            float v = xb[j];
            sumf += v;
            float av = v < 0 ? -v : v;
            if (av > amax) amax = av;
        }
#endif
        float d = amax / 127.0f;
        float id = (d > 0.0f) ? (127.0f / amax) : 0.0f;
        qx[b].d = d;
        qx[b].sumf = sumf;
#ifdef __ARM_NEON
        float32x4_t vid = vdupq_n_f32(id);
        float32x4_t vhalf = vdupq_n_f32(0.5f);
        float32x4_t vnhalf = vdupq_n_f32(-0.5f);
        float32x4_t vzero = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 8) {
            float32x4_t x0 = vmulq_f32(vld1q_f32(xb + j), vid);
            float32x4_t x1 = vmulq_f32(vld1q_f32(xb + j + 4), vid);
            uint32x4_t neg0 = vcltq_f32(x0, vzero);
            uint32x4_t neg1 = vcltq_f32(x1, vzero);
            x0 = vaddq_f32(x0, vbslq_f32(neg0, vnhalf, vhalf));
            x1 = vaddq_f32(x1, vbslq_f32(neg1, vnhalf, vhalf));
            int16x4_t i0 = vmovn_s32(vcvtq_s32_f32(x0));
            int16x4_t i1 = vmovn_s32(vcvtq_s32_f32(x1));
            int8x8_t q8 = vqmovn_s16(vcombine_s16(i0, i1));
            vst1_s8(qx[b].qs + j, q8);
        }
#else
        for (int j = 0; j < 32; j++) {
            float v = xb[j] * id;
            int iv = (int)(v + (v >= 0 ? 0.5f : -0.5f));
            if (iv > 127) iv = 127;
            if (iv < -127) iv = -127;
            qx[b].qs[j] = (int8_t)iv;
        }
#endif
    }
}

/* ---- Dot-product kernels (NEON + scalar) ---- */
static float dot_f32(const float *w, const float *x, uint32_t n) {
    float sum = 0.0f;
#ifdef __ARM_NEON
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __builtin_prefetch(w + i + 32, 0, 1);
        acc0 = vmlaq_f32(acc0, vld1q_f32(w + i), vld1q_f32(x + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(w + i + 4), vld1q_f32(x + i + 4));
    }
    acc0 = vaddq_f32(acc0, acc1);
    for (; i + 4 <= n; i += 4) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(w + i), vld1q_f32(x + i));
    }
    float32x2_t v2 = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
    sum = vget_lane_f32(vpadd_f32(v2, v2), 0);
    for (; i < n; i++) sum += w[i] * x[i];
#else
    for (uint32_t i = 0; i < n; i++) sum += w[i] * x[i];
#endif
    return sum;
}

static float dot_f16(const uint16_t *w, const float *x, uint32_t n) {
    float sum = 0.0f;
#ifdef __ARM_NEON
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __builtin_prefetch(w + i + 32, 0, 1);
        float w0[4] = { fp16_to_fp32(w[i]), fp16_to_fp32(w[i + 1]), fp16_to_fp32(w[i + 2]), fp16_to_fp32(w[i + 3]) };
        float w1[4] = { fp16_to_fp32(w[i + 4]), fp16_to_fp32(w[i + 5]), fp16_to_fp32(w[i + 6]), fp16_to_fp32(w[i + 7]) };
        acc0 = vmlaq_f32(acc0, vld1q_f32(w0), vld1q_f32(x + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(w1), vld1q_f32(x + i + 4));
    }
    acc0 = vaddq_f32(acc0, acc1);
    float32x2_t v2 = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
    sum = vget_lane_f32(vpadd_f32(v2, v2), 0);
    for (; i < n; i++) sum += fp16_to_fp32(w[i]) * x[i];
#else
    for (uint32_t i = 0; i < n; i++) sum += fp16_to_fp32(w[i]) * x[i];
#endif
    return sum;
}

static float dot_bf16(const uint16_t *w, const float *x, uint32_t n) {
    float sum = 0.0f;
#ifdef __ARM_NEON
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __builtin_prefetch(w + i + 32, 0, 1);
        uint16x4_t b0 = vld1_u16(w + i);
        uint16x4_t b1 = vld1_u16(w + i + 4);
        float32x4_t f0 = vreinterpretq_f32_u32(vshll_n_u16(b0, 16));
        float32x4_t f1 = vreinterpretq_f32_u32(vshll_n_u16(b1, 16));
        acc0 = vmlaq_f32(acc0, f0, vld1q_f32(x + i));
        acc1 = vmlaq_f32(acc1, f1, vld1q_f32(x + i + 4));
    }
    acc0 = vaddq_f32(acc0, acc1);
    float32x2_t v2 = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
    sum = vget_lane_f32(vpadd_f32(v2, v2), 0);
    for (; i < n; i++) sum += bf16_to_fp32(w[i]) * x[i];
#else
    for (uint32_t i = 0; i < n; i++) sum += bf16_to_fp32(w[i]) * x[i];
#endif
    return sum;
}

static float dot_q8_q8(const uint8_t *w, const BlockQ8 *qx, uint32_t n) {
    uint32_t nb = n / 32;
    float sum = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        __builtin_prefetch(w + 68, 0, 1);
        uint16_t d16; memcpy(&d16, w, 2);
        float d = fp16_to_fp32(d16);
        const int8_t *qs = (const int8_t *)(w + 2);
        const int8_t *qxs = qx[b].qs;
#ifdef __ARM_NEON
        int16x8_t p0 = vmull_s8(vld1_s8(qs),      vld1_s8(qxs));
        int16x8_t p1 = vmull_s8(vld1_s8(qs + 8),  vld1_s8(qxs + 8));
        int16x8_t p2 = vmull_s8(vld1_s8(qs + 16), vld1_s8(qxs + 16));
        int16x8_t p3 = vmull_s8(vld1_s8(qs + 24), vld1_s8(qxs + 24));
        int32x4_t acc = vpaddlq_s16(p0);
        acc = vpadalq_s16(acc, p1);
        acc = vpadalq_s16(acc, p2);
        acc = vpadalq_s16(acc, p3);
        int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
        int32_t isum = vget_lane_s32(vpadd_s32(s2, s2), 0);
#else
        int32_t isum = 0;
        for (int j = 0; j < 32; j++) isum += (int32_t)qs[j] * (int32_t)qxs[j];
#endif
        sum += d * qx[b].d * (float)isum;
        w += 34;
    }
    return sum;
}

static float dot_q4_0_q8(const uint8_t *w, const BlockQ8 *qx, uint32_t n) {
    uint32_t nb = n / 32;
    float sum = 0.0f;
#ifdef __ARM_NEON
    uint8x16_t vmask = vdupq_n_u8(0x0F);
    int8x16_t v8 = vdupq_n_s8(8);
#endif
    for (uint32_t b = 0; b < nb; b++) {
        __builtin_prefetch(w + 36, 0, 1);
        uint16_t d16; memcpy(&d16, w, 2);
        float d = fp16_to_fp32(d16);
        const uint8_t *qs = w + 2;
        const int8_t *qxs = qx[b].qs;
#ifdef __ARM_NEON
        uint8x16_t vq = vld1q_u8(qs);
        int8x16_t wlo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vq, vmask)), v8);
        int8x16_t whi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vq, 4)), v8);
        int8x16_t x0 = vld1q_s8(qxs);
        int8x16_t x1 = vld1q_s8(qxs + 16);
        int32x4_t acc = vpaddlq_s16(vmull_s8(vget_low_s8(wlo),  vget_low_s8(x0)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(wlo), vget_high_s8(x0)));
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(whi),  vget_low_s8(x1)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(whi), vget_high_s8(x1)));
        int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
        int32_t isum = vget_lane_s32(vpadd_s32(s2, s2), 0);
#else
        int32_t isum = 0;
        for (int j = 0; j < 16; j++) {
            isum += ((int)(qs[j] & 0x0F) - 8) * (int32_t)qxs[j];
            isum += ((int)(qs[j] >> 4)   - 8) * (int32_t)qxs[j + 16];
        }
#endif
        sum += d * qx[b].d * (float)isum;
        w += 18;
    }
    return sum;
}

static float dot_q4_1_q8(const uint8_t *w, const BlockQ8 *qx, uint32_t n) {
    uint32_t nb = n / 32;
    float sum = 0.0f;
#ifdef __ARM_NEON
    uint8x16_t vmask = vdupq_n_u8(0x0F);
#endif
    for (uint32_t b = 0; b < nb; b++) {
        __builtin_prefetch(w + 40, 0, 1);
        uint16_t d16, m16;
        memcpy(&d16, w, 2); memcpy(&m16, w + 2, 2);
        float d = fp16_to_fp32(d16), m = fp16_to_fp32(m16);
        const uint8_t *qs = w + 4;
        const int8_t *qxs = qx[b].qs;
#ifdef __ARM_NEON
        uint8x16_t vq = vld1q_u8(qs);
        int8x16_t wlo = vreinterpretq_s8_u8(vandq_u8(vq, vmask));
        int8x16_t whi = vreinterpretq_s8_u8(vshrq_n_u8(vq, 4));
        int8x16_t x0 = vld1q_s8(qxs);
        int8x16_t x1 = vld1q_s8(qxs + 16);
        int32x4_t acc = vpaddlq_s16(vmull_s8(vget_low_s8(wlo),  vget_low_s8(x0)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(wlo), vget_high_s8(x0)));
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(whi),  vget_low_s8(x1)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(whi), vget_high_s8(x1)));
        int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
        int32_t isum = vget_lane_s32(vpadd_s32(s2, s2), 0);
#else
        int32_t isum = 0;
        for (int j = 0; j < 16; j++) {
            isum += (int)(qs[j] & 0x0F) * (int32_t)qxs[j];
            isum += (int)(qs[j] >> 4)   * (int32_t)qxs[j + 16];
        }
#endif
        sum += d * qx[b].d * (float)isum + m * qx[b].sumf;
        w += 20;
    }
    return sum;
}

#ifdef __ARM_NEON
static const uint8_t k_q5_bitmask[16] = {
    1, 2, 4, 8, 16, 32, 64, 128,
    1, 2, 4, 8, 16, 32, 64, 128
};
#endif

static float dot_q5_0_q8(const uint8_t *w, const BlockQ8 *qx, uint32_t n) {
    uint32_t nb = n / 32;
    float sum = 0.0f;
#ifdef __ARM_NEON
    uint8x16_t vmask = vdupq_n_u8(0x0F);
    uint8x16_t vbitm = vld1q_u8(k_q5_bitmask);
    uint8x16_t v16 = vdupq_n_u8(0x10);
    int8x16_t vs16 = vdupq_n_s8(16);
#endif
    for (uint32_t b = 0; b < nb; b++) {
        __builtin_prefetch(w + 44, 0, 1);
        uint16_t d16; memcpy(&d16, w, 2);
        float d = fp16_to_fp32(d16);
        const uint8_t *qh_bytes = w + 2;
        const uint8_t *qs = w + 6;
        const int8_t *qxs = qx[b].qs;
#ifdef __ARM_NEON
        uint8x8_t qh_b0 = vdup_n_u8(qh_bytes[0]);
        uint8x8_t qh_b1 = vdup_n_u8(qh_bytes[1]);
        uint8x8_t qh_b2 = vdup_n_u8(qh_bytes[2]);
        uint8x8_t qh_b3 = vdup_n_u8(qh_bytes[3]);
        uint8x16_t qh_lo_src = vcombine_u8(qh_b0, qh_b1);
        uint8x16_t qh_hi_src = vcombine_u8(qh_b2, qh_b3);
        uint8x16_t h0 = vandq_u8(vtstq_u8(qh_lo_src, vbitm), v16);
        uint8x16_t h1 = vandq_u8(vtstq_u8(qh_hi_src, vbitm), v16);

        uint8x16_t vq = vld1q_u8(qs);
        int8x16_t wlo = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(vq, vmask), h0)), vs16);
        int8x16_t whi = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vq, 4), h1)), vs16);
        int8x16_t x0 = vld1q_s8(qxs);
        int8x16_t x1 = vld1q_s8(qxs + 16);
        int32x4_t acc = vpaddlq_s16(vmull_s8(vget_low_s8(wlo),  vget_low_s8(x0)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(wlo), vget_high_s8(x0)));
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(whi),  vget_low_s8(x1)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(whi), vget_high_s8(x1)));
        int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
        int32_t isum = vget_lane_s32(vpadd_s32(s2, s2), 0);
#else
        uint32_t qh; memcpy(&qh, qh_bytes, 4);
        int32_t isum = 0;
        for (int j = 0; j < 16; j++) {
            uint8_t xh_0 = ((qh >> j) << 4) & 0x10;
            uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
            isum += (((int)(qs[j] & 0x0F) | xh_0) - 16) * (int32_t)qxs[j];
            isum += (((int)(qs[j] >> 4)   | xh_1) - 16) * (int32_t)qxs[j + 16];
        }
#endif
        sum += d * qx[b].d * (float)isum;
        w += 22;
    }
    return sum;
}

static float dot_q5_1_q8(const uint8_t *w, const BlockQ8 *qx, uint32_t n) {
    uint32_t nb = n / 32;
    float sum = 0.0f;
#ifdef __ARM_NEON
    uint8x16_t vmask = vdupq_n_u8(0x0F);
    uint8x16_t vbitm = vld1q_u8(k_q5_bitmask);
    uint8x16_t v16 = vdupq_n_u8(0x10);
#endif
    for (uint32_t b = 0; b < nb; b++) {
        __builtin_prefetch(w + 48, 0, 1);
        uint16_t d16, m16;
        memcpy(&d16, w, 2); memcpy(&m16, w + 2, 2);
        float d = fp16_to_fp32(d16), m = fp16_to_fp32(m16);
        const uint8_t *qh_bytes = w + 4;
        const uint8_t *qs = w + 8;
        const int8_t *qxs = qx[b].qs;
#ifdef __ARM_NEON
        uint8x8_t qh_b0 = vdup_n_u8(qh_bytes[0]);
        uint8x8_t qh_b1 = vdup_n_u8(qh_bytes[1]);
        uint8x8_t qh_b2 = vdup_n_u8(qh_bytes[2]);
        uint8x8_t qh_b3 = vdup_n_u8(qh_bytes[3]);
        uint8x16_t qh_lo_src = vcombine_u8(qh_b0, qh_b1);
        uint8x16_t qh_hi_src = vcombine_u8(qh_b2, qh_b3);
        uint8x16_t h0 = vandq_u8(vtstq_u8(qh_lo_src, vbitm), v16);
        uint8x16_t h1 = vandq_u8(vtstq_u8(qh_hi_src, vbitm), v16);

        uint8x16_t vq = vld1q_u8(qs);
        int8x16_t wlo = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(vq, vmask), h0));
        int8x16_t whi = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vq, 4), h1));
        int8x16_t x0 = vld1q_s8(qxs);
        int8x16_t x1 = vld1q_s8(qxs + 16);
        int32x4_t acc = vpaddlq_s16(vmull_s8(vget_low_s8(wlo),  vget_low_s8(x0)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(wlo), vget_high_s8(x0)));
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(whi),  vget_low_s8(x1)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(whi), vget_high_s8(x1)));
        int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
        int32_t isum = vget_lane_s32(vpadd_s32(s2, s2), 0);
#else
        uint32_t qh; memcpy(&qh, qh_bytes, 4);
        int32_t isum = 0;
        for (int j = 0; j < 16; j++) {
            uint8_t xh_0 = ((qh >> j) << 4) & 0x10;
            uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
            isum += ((int)(qs[j] & 0x0F) | xh_0) * (int32_t)qxs[j];
            isum += ((int)(qs[j] >> 4)   | xh_1) * (int32_t)qxs[j + 16];
        }
#endif
        sum += d * qx[b].d * (float)isum + m * qx[b].sumf;
        w += 24;
    }
    return sum;
}

static float dot_q2_k_q8(const uint8_t *w, const BlockQ8 *qx, uint32_t n) {
    uint32_t nb = n / 256;
    float sum = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        __builtin_prefetch(w + 84, 0, 1);
        const uint8_t *scales = w;
        const uint8_t *qs = w + 16;
        uint16_t d16, dmin16;
        memcpy(&d16, w + 80, 2); memcpy(&dmin16, w + 82, 2);
        float d = fp16_to_fp32(d16), min = fp16_to_fp32(dmin16);
        const BlockQ8 *qb = qx + b * 8;
        int is = 0;
        for (int n128 = 0; n128 < 256; n128 += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                const BlockQ8 *qbj = qb + (n128 / 32) + j;
                uint8_t sc1 = scales[is++];
                uint8_t sc2 = scales[is++];
                int32_t isum1 = 0, isum2 = 0, sx1 = 0, sx2 = 0;
                for (int l = 0; l < 16; l++) {
                    int8_t x1 = qbj->qs[l], x2 = qbj->qs[l + 16];
                    isum1 += (int32_t)((qs[l]      >> shift) & 3) * (int32_t)x1;
                    isum2 += (int32_t)((qs[l + 16] >> shift) & 3) * (int32_t)x2;
                    sx1 += x1;
                    sx2 += x2;
                }
                sum += d * (float)(sc1 & 0xF) * qbj->d * (float)isum1 - min * (float)(sc1 >> 4) * qbj->d * (float)sx1;
                sum += d * (float)(sc2 & 0xF) * qbj->d * (float)isum2 - min * (float)(sc2 >> 4) * qbj->d * (float)sx2;
                shift += 2;
            }
            qs += 32;
        }
        w += 84;
    }
    return sum;
}

static float dot_q3_k_q8(const uint8_t *w, const BlockQ8 *qx, uint32_t n) {
    uint32_t nb = n / 256;
    float sum = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        __builtin_prefetch(w + 110, 0, 1);
        const uint8_t *hmask = w;
        const uint8_t *qs = w + 32;
        uint32_t aux[4];
        memcpy(aux, w + 96, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 4) & 0x0F0F0F0Fu) << 4);
        aux[3] = ((aux[1] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 0) & 0x0F0F0F0Fu) << 4);
        aux[0] = (aux[0] & 0x0F0F0F0Fu) | (((tmp >> 0) & 0x03030303u) << 4);
        aux[1] = (aux[1] & 0x0F0F0F0Fu) | (((tmp >> 2) & 0x03030303u) << 4);
        const int8_t *scales = (const int8_t *)aux;
        uint16_t d16; memcpy(&d16, w + 108, 2);
        float d_all = fp16_to_fp32(d16);
        const BlockQ8 *qb = qx + b * 8;
        int is = 0; uint8_t m = 1;
        for (int n128 = 0; n128 < 256; n128 += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                const BlockQ8 *qbj = qb + (n128 / 32) + j;
                int32_t isum1 = 0, isum2 = 0;
                for (int l = 0; l < 16; l++) {
                    int q1 = (int)((qs[l]      >> shift) & 3) - ((hmask[l]      & m) ? 0 : 4);
                    int q2 = (int)((qs[l + 16] >> shift) & 3) - ((hmask[l + 16] & m) ? 0 : 4);
                    isum1 += q1 * (int32_t)qbj->qs[l];
                    isum2 += q2 * (int32_t)qbj->qs[l + 16];
                }
                sum += d_all * (float)(scales[is++] - 32) * qbj->d * (float)isum1;
                sum += d_all * (float)(scales[is++] - 32) * qbj->d * (float)isum2;
                shift += 2; m <<= 1;
            }
            qs += 32;
        }
        w += 110;
    }
    return sum;
}

static float dot_q4_k_q8(const uint8_t *w, const BlockQ8 *qx, uint32_t n) {
    uint32_t nb = n / 256;
    float sum = 0.0f;
#ifdef __ARM_NEON
    uint8x16_t vmask = vdupq_n_u8(0x0F);
#endif
    for (uint32_t b = 0; b < nb; b++) {
        __builtin_prefetch(w + 144, 0, 1);
        uint16_t d16, dmin16;
        memcpy(&d16, w, 2); memcpy(&dmin16, w + 2, 2);
        float d = fp16_to_fp32(d16), dmin = fp16_to_fp32(dmin16);
        const uint8_t *scales = w + 4;
        const uint8_t *qs = w + 16;
        const BlockQ8 *qb = qx + b * 8;
        int is = 0;
        for (int c = 0; c < 4; c++) {
            uint8_t sc1, m1, sc2, m2;
            if (is < 4) { sc1 = scales[is] & 63; m1 = scales[is + 4] & 63; }
            else        { sc1 = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4); m1 = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4); }
            if (is + 1 < 4) { sc2 = scales[is + 1] & 63; m2 = scales[is + 5] & 63; }
            else            { sc2 = (scales[is + 5] & 0xF) | ((scales[is - 3] >> 6) << 4); m2 = (scales[is + 5] >> 4) | ((scales[is + 1] >> 6) << 4); }

            const int8_t *x0 = qb[2 * c].qs;
            const int8_t *x1 = qb[2 * c + 1].qs;
#ifdef __ARM_NEON
            uint8x16_t q0 = vld1q_u8(qs);
            uint8x16_t q1 = vld1q_u8(qs + 16);
            int8x16_t ql0 = vreinterpretq_s8_u8(vandq_u8(q0, vmask));
            int8x16_t ql1 = vreinterpretq_s8_u8(vandq_u8(q1, vmask));
            int8x16_t qh0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
            int8x16_t qh1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));

            int32x4_t a1 = vpaddlq_s16(vmull_s8(vget_low_s8(ql0),  vld1_s8(x0)));
            a1 = vpadalq_s16(a1, vmull_s8(vget_high_s8(ql0), vld1_s8(x0 + 8)));
            a1 = vpadalq_s16(a1, vmull_s8(vget_low_s8(ql1),  vld1_s8(x0 + 16)));
            a1 = vpadalq_s16(a1, vmull_s8(vget_high_s8(ql1), vld1_s8(x0 + 24)));
            int32x2_t s1 = vadd_s32(vget_low_s32(a1), vget_high_s32(a1));
            int32_t isum1 = vget_lane_s32(vpadd_s32(s1, s1), 0);

            int32x4_t a2 = vpaddlq_s16(vmull_s8(vget_low_s8(qh0),  vld1_s8(x1)));
            a2 = vpadalq_s16(a2, vmull_s8(vget_high_s8(qh0), vld1_s8(x1 + 8)));
            a2 = vpadalq_s16(a2, vmull_s8(vget_low_s8(qh1),  vld1_s8(x1 + 16)));
            a2 = vpadalq_s16(a2, vmull_s8(vget_high_s8(qh1), vld1_s8(x1 + 24)));
            int32x2_t s2 = vadd_s32(vget_low_s32(a2), vget_high_s32(a2));
            int32_t isum2 = vget_lane_s32(vpadd_s32(s2, s2), 0);
#else
            int32_t isum1 = 0, isum2 = 0;
            for (int l = 0; l < 32; l++) {
                isum1 += (int32_t)(qs[l] & 0xF) * (int32_t)x0[l];
                isum2 += (int32_t)(qs[l] >> 4)  * (int32_t)x1[l];
            }
#endif
            sum += d * sc1 * qb[2 * c].d * (float)isum1 - dmin * m1 * qb[2 * c].sumf;
            sum += d * sc2 * qb[2 * c + 1].d * (float)isum2 - dmin * m2 * qb[2 * c + 1].sumf;
            qs += 32;
            is += 2;
        }
        w += 144;
    }
    return sum;
}

static float dot_q5_k_q8(const uint8_t *w, const BlockQ8 *qx, uint32_t n) {
    uint32_t nb = n / 256;
    float sum = 0.0f;
#ifdef __ARM_NEON
    uint8x16_t vmask = vdupq_n_u8(0x0F);
    uint8x16_t v16 = vdupq_n_u8(16);
#endif
    for (uint32_t b = 0; b < nb; b++) {
        __builtin_prefetch(w + 176, 0, 1);
        uint16_t d16, dmin16;
        memcpy(&d16, w, 2); memcpy(&dmin16, w + 2, 2);
        float d = fp16_to_fp32(d16), dmin = fp16_to_fp32(dmin16);
        const uint8_t *scales = w + 4;
        const uint8_t *qh = w + 16;
        const uint8_t *qs = w + 48;
        const BlockQ8 *qb = qx + b * 8;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
#ifdef __ARM_NEON
        uint8x16_t vqh0 = vld1q_u8(qh);
        uint8x16_t vqh1 = vld1q_u8(qh + 16);
#endif
        for (int c = 0; c < 4; c++) {
            uint8_t sc1, m1, sc2, m2;
            if (is < 4) { sc1 = scales[is] & 63; m1 = scales[is + 4] & 63; }
            else        { sc1 = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4); m1 = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4); }
            if (is + 1 < 4) { sc2 = scales[is + 1] & 63; m2 = scales[is + 5] & 63; }
            else            { sc2 = (scales[is + 5] & 0xF) | ((scales[is - 3] >> 6) << 4); m2 = (scales[is + 5] >> 4) | ((scales[is + 1] >> 6) << 4); }

            const int8_t *x0 = qb[2 * c].qs;
            const int8_t *x1 = qb[2 * c + 1].qs;
#ifdef __ARM_NEON
            uint8x16_t vu1 = vdupq_n_u8(u1);
            uint8x16_t vu2 = vdupq_n_u8(u2);
            uint8x16_t q0 = vld1q_u8(qs);
            uint8x16_t q1 = vld1q_u8(qs + 16);
            uint8x16_t h1_0 = vandq_u8(vtstq_u8(vqh0, vu1), v16);
            uint8x16_t h1_1 = vandq_u8(vtstq_u8(vqh1, vu1), v16);
            uint8x16_t h2_0 = vandq_u8(vtstq_u8(vqh0, vu2), v16);
            uint8x16_t h2_1 = vandq_u8(vtstq_u8(vqh1, vu2), v16);

            int8x16_t wlo0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q0, vmask), h1_0));
            int8x16_t wlo1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q1, vmask), h1_1));
            int8x16_t whi0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q0, 4), h2_0));
            int8x16_t whi1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q1, 4), h2_1));

            int32x4_t a1 = vpaddlq_s16(vmull_s8(vget_low_s8(wlo0),  vld1_s8(x0)));
            a1 = vpadalq_s16(a1, vmull_s8(vget_high_s8(wlo0), vld1_s8(x0 + 8)));
            a1 = vpadalq_s16(a1, vmull_s8(vget_low_s8(wlo1),  vld1_s8(x0 + 16)));
            a1 = vpadalq_s16(a1, vmull_s8(vget_high_s8(wlo1), vld1_s8(x0 + 24)));
            int32x2_t s1 = vadd_s32(vget_low_s32(a1), vget_high_s32(a1));
            int32_t isum1 = vget_lane_s32(vpadd_s32(s1, s1), 0);

            int32x4_t a2 = vpaddlq_s16(vmull_s8(vget_low_s8(whi0),  vld1_s8(x1)));
            a2 = vpadalq_s16(a2, vmull_s8(vget_high_s8(whi0), vld1_s8(x1 + 8)));
            a2 = vpadalq_s16(a2, vmull_s8(vget_low_s8(whi1),  vld1_s8(x1 + 16)));
            a2 = vpadalq_s16(a2, vmull_s8(vget_high_s8(whi1), vld1_s8(x1 + 24)));
            int32x2_t s2 = vadd_s32(vget_low_s32(a2), vget_high_s32(a2));
            int32_t isum2 = vget_lane_s32(vpadd_s32(s2, s2), 0);
#else
            int32_t isum1 = 0, isum2 = 0;
            for (int l = 0; l < 32; l++) {
                isum1 += (int32_t)((qs[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) * (int32_t)x0[l];
                isum2 += (int32_t)((qs[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) * (int32_t)x1[l];
            }
#endif
            sum += d * sc1 * qb[2 * c].d * (float)isum1 - dmin * m1 * qb[2 * c].sumf;
            sum += d * sc2 * qb[2 * c + 1].d * (float)isum2 - dmin * m2 * qb[2 * c + 1].sumf;
            qs += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        w += 176;
    }
    return sum;
}

static float dot_q6_k_q8(const uint8_t *w, const BlockQ8 *qx, uint32_t n) {
    uint32_t nb = n / 256;
    float sum = 0.0f;
#ifdef __ARM_NEON
    uint8x16_t vmask4 = vdupq_n_u8(0x0F);
    uint8x16_t vmask2 = vdupq_n_u8(0x03);
    int8x16_t v32 = vdupq_n_s8(32);
#endif
    for (uint32_t b = 0; b < nb; b++) {
        __builtin_prefetch(w + 210, 0, 1);
        const uint8_t *ql = w;
        const uint8_t *qh = w + 128;
        const int8_t  *sc = (const int8_t *)(w + 192);
        uint16_t d16; memcpy(&d16, w + 208, 2);
        float d = fp16_to_fp32(d16);
        const BlockQ8 *qb = qx + b * 8;

        for (int h = 0; h < 2; h++) {
            const BlockQ8 *qbh = qb + h * 4;
#ifdef __ARM_NEON
            for (int k = 0; k < 2; k++) {
                uint8x16_t ql0 = vld1q_u8(ql + k * 16);
                uint8x16_t ql1 = vld1q_u8(ql + 32 + k * 16);
                uint8x16_t qhh = vld1q_u8(qh + k * 16);

                int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, vmask4), vshlq_n_u8(vandq_u8(qhh, vmask2), 4))), v32);
                int8x16_t q2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, vmask4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhh, 2), vmask2), 4))), v32);
                int8x16_t q3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0, 4),    vshlq_n_u8(vandq_u8(vshrq_n_u8(qhh, 4), vmask2), 4))), v32);
                int8x16_t q4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1, 4),    vshlq_n_u8(vandq_u8(vshrq_n_u8(qhh, 6), vmask2), 4))), v32);

                const int8_t *x0 = qbh[0].qs + k * 16;
                const int8_t *x1 = qbh[1].qs + k * 16;
                const int8_t *x2 = qbh[2].qs + k * 16;
                const int8_t *x3 = qbh[3].qs + k * 16;

                int32x4_t a0 = vpaddlq_s16(vmull_s8(vget_low_s8(q1), vld1_s8(x0)));
                a0 = vpadalq_s16(a0, vmull_s8(vget_high_s8(q1), vld1_s8(x0 + 8)));
                int32x2_t s0 = vadd_s32(vget_low_s32(a0), vget_high_s32(a0));
                int32_t i0 = vget_lane_s32(vpadd_s32(s0, s0), 0);

                int32x4_t a1 = vpaddlq_s16(vmull_s8(vget_low_s8(q2), vld1_s8(x1)));
                a1 = vpadalq_s16(a1, vmull_s8(vget_high_s8(q2), vld1_s8(x1 + 8)));
                int32x2_t s1 = vadd_s32(vget_low_s32(a1), vget_high_s32(a1));
                int32_t i1 = vget_lane_s32(vpadd_s32(s1, s1), 0);

                int32x4_t a2 = vpaddlq_s16(vmull_s8(vget_low_s8(q3), vld1_s8(x2)));
                a2 = vpadalq_s16(a2, vmull_s8(vget_high_s8(q3), vld1_s8(x2 + 8)));
                int32x2_t s2 = vadd_s32(vget_low_s32(a2), vget_high_s32(a2));
                int32_t i2 = vget_lane_s32(vpadd_s32(s2, s2), 0);

                int32x4_t a3 = vpaddlq_s16(vmull_s8(vget_low_s8(q4), vld1_s8(x3)));
                a3 = vpadalq_s16(a3, vmull_s8(vget_high_s8(q4), vld1_s8(x3 + 8)));
                int32x2_t s3 = vadd_s32(vget_low_s32(a3), vget_high_s32(a3));
                int32_t i3 = vget_lane_s32(vpadd_s32(s3, s3), 0);

                sum += d * (
                    (float)sc[k + 0] * qbh[0].d * (float)i0 +
                    (float)sc[k + 2] * qbh[1].d * (float)i1 +
                    (float)sc[k + 4] * qbh[2].d * (float)i2 +
                    (float)sc[k + 6] * qbh[3].d * (float)i3
                );
            }
#else
            for (int k = 0; k < 2; k++) {
                int32_t i0 = 0, i1 = 0, i2 = 0, i3 = 0;
                for (int l = k * 16; l < k * 16 + 16; l++) {
                    int8_t q1 = (int8_t)((ql[l + 0]  & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l + 0]  >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                    i0 += (int32_t)q1 * (int32_t)qbh[0].qs[l];
                    i1 += (int32_t)q2 * (int32_t)qbh[1].qs[l];
                    i2 += (int32_t)q3 * (int32_t)qbh[2].qs[l];
                    i3 += (int32_t)q4 * (int32_t)qbh[3].qs[l];
                }
                sum += d * (
                    (float)sc[k + 0] * qbh[0].d * (float)i0 +
                    (float)sc[k + 2] * qbh[1].d * (float)i1 +
                    (float)sc[k + 4] * qbh[2].d * (float)i2 +
                    (float)sc[k + 6] * qbh[3].d * (float)i3
                );
            }
#endif
            ql += 64; qh += 32; sc += 8;
        }
        w += 210;
    }
    return sum;
}

static float dot_row(uint32_t type, const uint8_t *w, const float *x, const BlockQ8 *qx, uint32_t n) {
    switch (type) {
        case GGML_T_F32:  return dot_f32((const float *)w, x, n);
        case GGML_T_F16:  return dot_f16((const uint16_t *)w, x, n);
        case GGML_T_BF16: return dot_bf16((const uint16_t *)w, x, n);
        case GGML_T_Q8_0: return dot_q8_q8(w, qx, n);
        case GGML_T_Q4_0: return dot_q4_0_q8(w, qx, n);
        case GGML_T_Q4_1: return dot_q4_1_q8(w, qx, n);
        case GGML_T_Q5_0: return dot_q5_0_q8(w, qx, n);
        case GGML_T_Q5_1: return dot_q5_1_q8(w, qx, n);
        case GGML_T_Q2_K: return dot_q2_k_q8(w, qx, n);
        case GGML_T_Q3_K: return dot_q3_k_q8(w, qx, n);
        case GGML_T_Q4_K: return dot_q4_k_q8(w, qx, n);
        case GGML_T_Q5_K: return dot_q5_k_q8(w, qx, n);
        case GGML_T_Q6_K: return dot_q6_k_q8(w, qx, n);
        default: return 0.0f;
    }
}

/* ---- Context & Weights ---- */
typedef struct {
    int id;
    float prob;
} ProbIdx;

struct IndextContext {
    IndextParams params;
    IndextModelInfo info;
    IndextStats stats;

#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMap;
#else
    int fd;
#endif
    uint8_t *mapped_ptr;
    size_t mapped_size;
    int used_ram_buf;

    GgufFile *gguf;
    IndextTokenizer *tok;

    struct {
        TensorView tok_emb;
        TensorView lm_head;
        float *norm_final;
        float *norm_final_b;
        struct {
            float *attn_norm;
            float *attn_norm_b;
            float *post_attn_norm;
            float *ffn_norm;
            float *ffn_norm_b;
            float *post_ffn_norm;
            float *q_norm;
            float *k_norm;
            TensorView wq, wk, wv, wo, wqkv;
            int has_wqkv;
            float *bq, *bk, *bv, *bo;
            TensorView w1, w2, w3;
            int has_w1;
            float *b1, *b2, *b3;
        } *layers;
    } w;
    int use_gelu;
    float emb_scale;
    float attn_softcap;
    float final_softcap;

    float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *att, *logits;
    float *key_cache, *val_cache;
    uint16_t *key_cache_f16, *val_cache_f16;
    int8_t *key_cache_q8, *val_cache_q8;
    float *key_scale_q8, *val_scale_q8;
    uint8_t *key_cache_q4, *val_cache_q4;
    float *key_scale_q4, *val_scale_q4;
    int *recent_tokens;
    int n_recent;
    int cur_pos;
    uint32_t rope_dim;
    int rope_neox;
    volatile int stop_requested;
    uint64_t rng_state;

    BlockQ8 *qx_buf;
    int qx_valid;
    float *rope_inv_freq;
    float *rope_cos;
    float *rope_sin;
    ProbIdx *probs_buf;

#ifndef _WIN32
    pthread_t pool_threads[16];
    int pool_n_workers;
    pthread_mutex_t pool_mtx;
    pthread_cond_t pool_cond_work;
    pthread_cond_t pool_cond_done;
    volatile int pool_gen;
    volatile int pool_shutdown;
    volatile int pool_active_workers;
    volatile int pool_done_workers;
    const TensorView *pool_mat;
    const float *pool_x;
    const BlockQ8 *pool_qx;
    float *pool_out;
    size_t pool_rb;
    uint32_t pool_r0[16];
    uint32_t pool_r1[16];
#endif
};

static void run_matmul_slice(const TensorView *mat, const float *x, const BlockQ8 *qx, float *out, uint32_t r0, uint32_t r1, size_t rb) {
    uint32_t cols = mat->cols;
    uint32_t type = mat->type;
    const uint8_t *row_ptr = mat->data + (size_t)r0 * rb;
    uint32_t r = r0;
    for (; r + 1 < r1; r += 2) {
        const uint8_t *row0 = row_ptr;
        const uint8_t *row1 = row_ptr + rb;
        __builtin_prefetch(row1 + rb, 0, 1);
        out[r]     = dot_row(type, row0, x, qx, cols);
        out[r + 1] = dot_row(type, row1, x, qx, cols);
        row_ptr += rb * 2;
    }
    if (r < r1) {
        out[r] = dot_row(type, row_ptr, x, qx, cols);
    }
}

#ifdef _WIN32
static void matmul_mt_ex(IndextContext *ctx, const TensorView *mat, const float *x, float *out, int n_threads, int prequant) {
    (void)n_threads;
    const BlockQ8 *qx = NULL;
    if (needs_q8_x(mat->type)) {
        if (prequant || !ctx->qx_valid) {
            quantize_x_q8(x, ctx->qx_buf, mat->cols);
            ctx->qx_valid = 1;
        }
        qx = ctx->qx_buf;
    }
    run_matmul_slice(mat, x, qx, out, 0, mat->rows, tensor_row_bytes(mat->type, mat->cols));
}
#else
typedef struct {
    IndextContext *ctx;
    int worker_id;
} PoolWorkerArg;

static PoolWorkerArg g_worker_args[16];

static void *pool_worker_posix(void *arg) {
    PoolWorkerArg *wa = (PoolWorkerArg *)arg;
    IndextContext *ctx = wa->ctx;
    int id = wa->worker_id;
    int seen_gen = 0;

    for (;;) {
        pthread_mutex_lock(&ctx->pool_mtx);
        while (!ctx->pool_shutdown && ctx->pool_gen == seen_gen) {
            pthread_cond_wait(&ctx->pool_cond_work, &ctx->pool_mtx);
        }
        if (ctx->pool_shutdown) {
            pthread_mutex_unlock(&ctx->pool_mtx);
            return NULL;
        }
        seen_gen = ctx->pool_gen;
        int active = ctx->pool_active_workers;
        uint32_t r0 = ctx->pool_r0[id];
        uint32_t r1 = ctx->pool_r1[id];
        const TensorView *mat = ctx->pool_mat;
        const float *x = ctx->pool_x;
        const BlockQ8 *qx = ctx->pool_qx;
        float *out = ctx->pool_out;
        size_t rb = ctx->pool_rb;
        pthread_mutex_unlock(&ctx->pool_mtx);

        if (id <= active && r0 < r1) {
            run_matmul_slice(mat, x, qx, out, r0, r1, rb);
        }

        if (id <= active) {
            pthread_mutex_lock(&ctx->pool_mtx);
            ctx->pool_done_workers++;
            if (ctx->pool_done_workers >= active) {
                pthread_cond_signal(&ctx->pool_cond_done);
            }
            pthread_mutex_unlock(&ctx->pool_mtx);
        }
    }
}

static void pool_init(IndextContext *ctx, int n_threads) {
    if (n_threads > 16) n_threads = 16;
    if (n_threads < 1) n_threads = 1;
    pthread_mutex_init(&ctx->pool_mtx, NULL);
    pthread_cond_init(&ctx->pool_cond_work, NULL);
    pthread_cond_init(&ctx->pool_cond_done, NULL);
    ctx->pool_n_workers = n_threads - 1;
    for (int i = 1; i <= ctx->pool_n_workers; i++) {
        g_worker_args[i].ctx = ctx;
        g_worker_args[i].worker_id = i;
        pthread_create(&ctx->pool_threads[i - 1], NULL, pool_worker_posix, &g_worker_args[i]);
    }
}

static void pool_destroy(IndextContext *ctx) {
    if (ctx->pool_n_workers > 0) {
        pthread_mutex_lock(&ctx->pool_mtx);
        ctx->pool_shutdown = 1;
        pthread_cond_broadcast(&ctx->pool_cond_work);
        pthread_mutex_unlock(&ctx->pool_mtx);
        for (int i = 0; i < ctx->pool_n_workers; i++) {
            pthread_join(ctx->pool_threads[i], NULL);
        }
    }
    pthread_mutex_destroy(&ctx->pool_mtx);
    pthread_cond_destroy(&ctx->pool_cond_work);
    pthread_cond_destroy(&ctx->pool_cond_done);
}

static void matmul_mt_ex(IndextContext *ctx, const TensorView *mat, const float *x, float *out, int n_threads, int prequant) {
    uint32_t rows = mat->rows;
    size_t rb = tensor_row_bytes(mat->type, mat->cols);
    const BlockQ8 *qx = NULL;
    if (needs_q8_x(mat->type)) {
        if (prequant || !ctx->qx_valid) {
            quantize_x_q8(x, ctx->qx_buf, mat->cols);
            ctx->qx_valid = 1;
        }
        qx = ctx->qx_buf;
    }
    int max_t = ctx->pool_n_workers + 1;
    if (n_threads > max_t) n_threads = max_t;
    if (n_threads <= 1 || rows < 64) {
        run_matmul_slice(mat, x, qx, out, 0, rows, rb);
        return;
    }
    if ((uint32_t)n_threads > rows) n_threads = (int)rows;

    uint32_t chunk = rows / (uint32_t)n_threads;
    uint32_t rem = rows % (uint32_t)n_threads;
    uint32_t cur = 0;

    pthread_mutex_lock(&ctx->pool_mtx);
    for (int i = 0; i < n_threads; i++) {
        uint32_t len = chunk + ((uint32_t)i < rem ? 1 : 0);
        ctx->pool_r0[i] = cur;
        ctx->pool_r1[i] = cur + len;
        cur += len;
    }
    ctx->pool_mat = mat;
    ctx->pool_x = x;
    ctx->pool_qx = qx;
    ctx->pool_out = out;
    ctx->pool_rb = rb;
    ctx->pool_active_workers = n_threads - 1;
    ctx->pool_done_workers = 0;
    ctx->pool_gen++;
    pthread_cond_broadcast(&ctx->pool_cond_work);
    pthread_mutex_unlock(&ctx->pool_mtx);

    run_matmul_slice(mat, x, qx, out, ctx->pool_r0[0], ctx->pool_r1[0], rb);

    pthread_mutex_lock(&ctx->pool_mtx);
    while (ctx->pool_done_workers < ctx->pool_active_workers) {
        pthread_cond_wait(&ctx->pool_cond_done, &ctx->pool_mtx);
    }
    pthread_mutex_unlock(&ctx->pool_mtx);
}
#endif

/* ---- Math ops ---- */
static void rmsnorm(float *o, const float *x, const float *w, uint32_t n, float eps) {
    float ss = 0.0f;
#ifdef __ARM_NEON
    float32x4_t vss = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        vss = vmlaq_f32(vss, vx, vx);
    }
    float32x2_t v2 = vadd_f32(vget_low_f32(vss), vget_high_f32(vss));
    ss = vget_lane_f32(vpadd_f32(v2, v2), 0);
    for (; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)n + eps);
    float32x4_t vscale = vdupq_n_f32(ss);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vw = vld1q_f32(w + i);
        vst1q_f32(o + i, vmulq_f32(vw, vmulq_f32(vscale, vx)));
    }
    for (; i < n; i++) o[i] = w[i] * (ss * x[i]);
#else
    for (uint32_t i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)n + eps);
    for (uint32_t i = 0; i < n; i++) o[i] = w[i] * (ss * x[i]);
#endif
}

static void layernorm(float *o, const float *x, const float *w, const float *b, uint32_t n, float eps) {
    float mean = 0.0f;
    for (uint32_t i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;
    float var = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float inv = 1.0f / sqrtf(var / (float)n + eps);
    for (uint32_t i = 0; i < n; i++) {
        float v = (x[i] - mean) * inv * w[i];
        o[i] = b ? (v + b[i]) : v;
    }
}

static inline void vec_add_f32(float *dst, const float *src, uint32_t n) {
    uint32_t i = 0;
#ifdef __ARM_NEON
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(dst + i, vaddq_f32(vld1q_f32(dst + i), vld1q_f32(src + i)));
    }
#endif
    for (; i < n; i++) dst[i] += src[i];
}

static inline float gelu_f32(float v) {
    return 0.5f * v * (1.0f + tanhf(0.7978845608f * v * (1.0f + 0.044715f * v * v)));
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = fast_expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < size; i++) x[i] *= inv;
}

static void apply_rope_cached(float *vec, uint32_t n_heads, uint32_t head_dim, uint32_t rope_dim, const float *rcos, const float *rsin, int neox) {
    uint32_t half = rope_dim / 2;
    for (uint32_t h = 0; h < n_heads; h++) {
        float *vh = vec + h * head_dim;
        for (uint32_t i = 0; i < half; i++) {
            float fcr = rcos[i];
            float fci = rsin[i];
            uint32_t i0 = neox ? i : (2 * i);
            uint32_t i1 = neox ? (i + half) : (2 * i + 1);
            float v0 = vh[i0];
            float v1 = vh[i1];
            vh[i0] = v0 * fcr - v1 * fci;
            vh[i1] = v0 * fci + v1 * fcr;
        }
    }
}

/* ---- File mapping / RAM loading helpers ---- */
static int map_file_ro(IndextContext *ctx, const char *path, char *err_buf, size_t err_len) {
#ifdef _WIN32
    ctx->hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (ctx->hFile == INVALID_HANDLE_VALUE) {
        snprintf(err_buf, err_len, "Cannot open file: %s", path);
        return -1;
    }
    LARGE_INTEGER fsz;
    GetFileSizeEx(ctx->hFile, &fsz);
    ctx->mapped_size = (size_t)fsz.QuadPart;
    ctx->hMap = CreateFileMappingA(ctx->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!ctx->hMap) {
        CloseHandle(ctx->hFile);
        ctx->hFile = INVALID_HANDLE_VALUE;
        snprintf(err_buf, err_len, "CreateFileMapping failed");
        return -1;
    }
    ctx->mapped_ptr = (uint8_t *)MapViewOfFile(ctx->hMap, FILE_MAP_READ, 0, 0, 0);
    if (!ctx->mapped_ptr) {
        CloseHandle(ctx->hMap); CloseHandle(ctx->hFile);
        ctx->hMap = NULL; ctx->hFile = INVALID_HANDLE_VALUE;
        snprintf(err_buf, err_len, "MapViewOfFile failed");
        return -1;
    }
    return 0;
#else
    ctx->fd = open(path, O_RDONLY);
    if (ctx->fd < 0) {
        snprintf(err_buf, err_len, "Cannot open file: %s", path);
        return -1;
    }
    struct stat st;
    if (fstat(ctx->fd, &st) != 0 || st.st_size <= 0) {
        close(ctx->fd); ctx->fd = -1;
        snprintf(err_buf, err_len, "Cannot stat file");
        return -1;
    }
    ctx->mapped_size = (size_t)st.st_size;
    void *p = mmap(NULL, ctx->mapped_size, PROT_READ, MAP_PRIVATE, ctx->fd, 0);
    if (p == MAP_FAILED) {
        close(ctx->fd); ctx->fd = -1;
        snprintf(err_buf, err_len, "mmap failed (file too large for 32-bit address space?)");
        return -1;
    }
    ctx->mapped_ptr = (uint8_t *)p;
    return 0;
#endif
}

static int load_file_ram(IndextContext *ctx, const char *path, char *err_buf, size_t err_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(err_buf, err_len, "Cannot open file: %s", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsz <= 0) {
        fclose(f);
        snprintf(err_buf, err_len, "Empty or unreadable model file");
        return -1;
    }
    uint8_t *buf = (uint8_t *)malloc((size_t)fsz);
    if (!buf) {
        fclose(f);
        snprintf(err_buf, err_len, "Out of RAM reading %.1f MB model (enable mmap in Model Settings)", (double)fsz / (1024.0 * 1024.0));
        return -1;
    }
    size_t off = 0;
    const size_t chunk = 4u * 1024u * 1024u;
    while (off < (size_t)fsz) {
        size_t todo = (size_t)fsz - off;
        if (todo > chunk) todo = chunk;
        size_t nr = fread(buf + off, 1, todo, f);
        if (nr == 0) break;
        off += nr;
    }
    fclose(f);
    if (off != (size_t)fsz) {
        free(buf);
        snprintf(err_buf, err_len, "Truncated read from model file");
        return -1;
    }
    ctx->mapped_ptr = buf;
    ctx->mapped_size = (size_t)fsz;
    ctx->used_ram_buf = 1;
    return 0;
}

static void unmap_file_ro(IndextContext *ctx) {
    if (ctx->used_ram_buf) {
        free(ctx->mapped_ptr);
        ctx->mapped_ptr = NULL;
        return;
    }
#ifdef _WIN32
    if (ctx->mapped_ptr) UnmapViewOfFile(ctx->mapped_ptr);
    if (ctx->hMap) CloseHandle(ctx->hMap);
    if (ctx->hFile && ctx->hFile != INVALID_HANDLE_VALUE) CloseHandle(ctx->hFile);
#else
    if (ctx->mapped_ptr) munmap(ctx->mapped_ptr, ctx->mapped_size);
    if (ctx->fd >= 0) close(ctx->fd);
#endif
}

static float *unpack_1d_tensor(const TensorView *tv) {
    float *buf = (float *)malloc((size_t)tv->cols * sizeof(float));
    if (buf) dequant_row(tv, 0, buf);
    return buf;
}

static const char *quant_type_name(uint32_t t) {
    switch (t) {
        case GGML_T_F32:  return "F32";
        case GGML_T_F16:  return "F16";
        case GGML_T_BF16: return "BF16";
        case GGML_T_Q4_0: return "Q4_0";
        case GGML_T_Q4_1: return "Q4_1";
        case GGML_T_Q5_0: return "Q5_0";
        case GGML_T_Q5_1: return "Q5_1";
        case GGML_T_Q8_0: return "Q8_0";
        case GGML_T_Q2_K: return "Q2_K";
        case GGML_T_Q3_K: return "Q3_K";
        case GGML_T_Q4_K: return "Q4_K";
        case GGML_T_Q5_K: return "Q5_K";
        case GGML_T_Q6_K: return "Q6_K";
        default:          return "UNKNOWN";
    }
}

static int load_idxt(IndextContext *ctx, const char *tok_path, char *err_buf, size_t err_len) {
    if (ctx->mapped_size < 28) {
        snprintf(err_buf, err_len, "IDXT file too small");
        return -1;
    }
    if (!tok_path || !tok_path[0]) {
        snprintf(err_buf, err_len, "IDXT format requires a tokenizer.json file");
        return -1;
    }
    ctx->tok = tok_load_json(tok_path, err_buf, err_len);
    if (!ctx->tok) return -1;

    uint32_t hdr[6];
    memcpy(hdr, ctx->mapped_ptr + 4, 24);
    if (hdr[0] == 0 || hdr[1] == 0 || hdr[2] == 0 || hdr[3] == 0 || hdr[4] == 0 || hdr[5] == 0) {
        snprintf(err_buf, err_len, "Invalid IDXT header (zero dimension)");
        return -1;
    }
    if ((hdr[1] % 32) != 0 || (hdr[4] % 32) != 0 || (hdr[1] % hdr[3]) != 0) {
        snprintf(err_buf, err_len, "Invalid IDXT dimensions (must align to 32 and n_heads)");
        return -1;
    }
    ctx->info.format = INDEXT_FMT_IDXT;
    ctx->info.idxt_version = 1;
    ctx->info.quant_type = GGML_T_Q8_0;
    snprintf(ctx->info.arch, sizeof(ctx->info.arch), "llama (IDXT Q8_0)");
    snprintf(ctx->info.model_name, sizeof(ctx->info.model_name), "InDEXt-IDXT");
    ctx->info.vocab_size  = hdr[0];
    ctx->info.dim         = hdr[1];
    ctx->info.n_layers    = hdr[2];
    ctx->info.n_heads     = hdr[3];
    ctx->info.n_kv_heads  = hdr[3];
    ctx->info.hidden_dim  = hdr[4];
    ctx->info.max_seq_len = hdr[5];
    ctx->info.head_dim    = ctx->info.dim / ctx->info.n_heads;
    ctx->info.rms_eps     = 1e-5f;
    ctx->info.rope_freq_base = 10000.0f;
    ctx->info.bos_id      = tok_bos_id(ctx->tok);
    ctx->info.eos_id      = tok_eos_id(ctx->tok);
    ctx->info.add_bos_default = 1;
    ctx->info.is_chatml   = 0;
    ctx->rope_neox        = 1;

    uint32_t V = ctx->info.vocab_size;
    uint32_t D = ctx->info.dim;
    uint32_t H = ctx->info.hidden_dim;
    uint32_t L = ctx->info.n_layers;

    const uint8_t *p = ctx->mapped_ptr + 28;
    const uint8_t *end = ctx->mapped_ptr + ctx->mapped_size;

    ctx->w.tok_emb = (TensorView){ p, GGML_T_Q8_0, V, D };
    p += (size_t)V * tensor_row_bytes(GGML_T_Q8_0, D);
    ctx->w.lm_head = ctx->w.tok_emb;

    ctx->w.layers = calloc(L, sizeof(*ctx->w.layers));
    for (uint32_t i = 0; i < L; i++) {
        if (p >= end) { snprintf(err_buf, err_len, "Truncated IDXT file"); return -1; }
        TensorView an = { p, GGML_T_Q8_0, 1, D };
        p += tensor_row_bytes(GGML_T_Q8_0, D);
        ctx->w.layers[i].attn_norm = unpack_1d_tensor(&an);

        ctx->w.layers[i].wq = (TensorView){ p, GGML_T_Q8_0, D, D }; p += (size_t)D * tensor_row_bytes(GGML_T_Q8_0, D);
        ctx->w.layers[i].wk = (TensorView){ p, GGML_T_Q8_0, D, D }; p += (size_t)D * tensor_row_bytes(GGML_T_Q8_0, D);
        ctx->w.layers[i].wv = (TensorView){ p, GGML_T_Q8_0, D, D }; p += (size_t)D * tensor_row_bytes(GGML_T_Q8_0, D);
        ctx->w.layers[i].wo = (TensorView){ p, GGML_T_Q8_0, D, D }; p += (size_t)D * tensor_row_bytes(GGML_T_Q8_0, D);

        TensorView fn = { p, GGML_T_Q8_0, 1, D };
        p += tensor_row_bytes(GGML_T_Q8_0, D);
        ctx->w.layers[i].ffn_norm = unpack_1d_tensor(&fn);

        ctx->w.layers[i].w1 = (TensorView){ p, GGML_T_Q8_0, H, D }; p += (size_t)H * tensor_row_bytes(GGML_T_Q8_0, D);
        ctx->w.layers[i].w2 = (TensorView){ p, GGML_T_Q8_0, D, H }; p += (size_t)D * tensor_row_bytes(GGML_T_Q8_0, H);
        ctx->w.layers[i].w3 = (TensorView){ p, GGML_T_Q8_0, H, D }; p += (size_t)H * tensor_row_bytes(GGML_T_Q8_0, D);
        ctx->w.layers[i].has_w1 = 1;
    }
    ctx->emb_scale = 1.0f;
    TensorView nf = { p, GGML_T_Q8_0, 1, D };
    p += tensor_row_bytes(GGML_T_Q8_0, D);
    if (p > end) {
        snprintf(err_buf, err_len, "Truncated IDXT file");
        return -1;
    }
    ctx->w.norm_final = unpack_1d_tensor(&nf);
    return 0;
}

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t vocab_size;
    uint32_t dim;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t hidden_dim;
    uint32_t max_seq_len;
    uint32_t head_dim;
    uint32_t quant_type;
    uint32_t flags;
    float rms_eps;
    float rope_freq_base;
    uint32_t reserved[2];
} Idxt2Header;

static int load_idxt2(IndextContext *ctx, const char *tok_path, char *err_buf, size_t err_len) {
    if (ctx->mapped_size < sizeof(Idxt2Header)) {
        snprintf(err_buf, err_len, "IDX2 file too small");
        return -1;
    }
    if (!tok_path || !tok_path[0]) {
        snprintf(err_buf, err_len, "IDXT v2 format requires a tokenizer.json file");
        return -1;
    }
    Idxt2Header hdr;
    memcpy(&hdr, ctx->mapped_ptr, sizeof(hdr));
    if (hdr.version != 2) {
        snprintf(err_buf, err_len, "Unsupported IDX2 version: %u", hdr.version);
        return -1;
    }
    if (!is_supported_quant_type(hdr.quant_type)) {
        snprintf(err_buf, err_len, "Unsupported IDX2 quant type: %u", hdr.quant_type);
        return -1;
    }
    if (hdr.vocab_size == 0 || hdr.dim == 0 || hdr.n_layers == 0 || hdr.n_heads == 0 || hdr.hidden_dim == 0 || hdr.max_seq_len == 0) {
        snprintf(err_buf, err_len, "Invalid IDX2 header dimensions");
        return -1;
    }

    ctx->tok = tok_load_json(tok_path, err_buf, err_len);
    if (!ctx->tok) return -1;

    uint32_t qt = hdr.quant_type;
    uint32_t n_kv = hdr.n_kv_heads ? hdr.n_kv_heads : hdr.n_heads;
    if (n_kv == 0 || n_kv > hdr.n_heads || (hdr.n_heads % n_kv) != 0) {
        snprintf(err_buf, err_len, "Invalid IDX2 n_kv_heads (%u vs %u)", n_kv, hdr.n_heads);
        return -1;
    }
    uint32_t hd = hdr.head_dim ? hdr.head_dim : (hdr.dim / hdr.n_heads);
    uint32_t Q_D = hdr.n_heads * hd;
    uint32_t KV_D = n_kv * hd;
    uint32_t bs = (qt == GGML_T_Q4_K || qt == GGML_T_Q5_K || qt == GGML_T_Q6_K) ? 256u
                : ((qt == GGML_T_F32 || qt == GGML_T_F16 || qt == GGML_T_BF16) ? 1u : 32u);
    if ((hdr.dim % bs) != 0 || (hdr.hidden_dim % bs) != 0 || (Q_D % bs) != 0 || (KV_D % bs) != 0) {
        snprintf(err_buf, err_len, "IDX2 dimensions not aligned to quant block size %u", bs);
        return -1;
    }
    int tied = (hdr.flags & 1u) != 0;

    ctx->info.format = INDEXT_FMT_IDXT;
    ctx->info.idxt_version = 2;
    ctx->info.quant_type = qt;
    snprintf(ctx->info.arch, sizeof(ctx->info.arch), "llama (IDX2 %s)", quant_type_name(qt));
    snprintf(ctx->info.model_name, sizeof(ctx->info.model_name), "InDEXt-IDXTv2");
    ctx->info.vocab_size  = hdr.vocab_size;
    ctx->info.dim         = hdr.dim;
    ctx->info.n_layers    = hdr.n_layers;
    ctx->info.n_heads     = hdr.n_heads;
    ctx->info.n_kv_heads  = n_kv;
    ctx->info.hidden_dim  = hdr.hidden_dim;
    ctx->info.max_seq_len = hdr.max_seq_len;
    ctx->info.head_dim    = hd;
    ctx->info.rms_eps     = (hdr.rms_eps > 0.0f) ? hdr.rms_eps : 1e-5f;
    ctx->info.rope_freq_base = (hdr.rope_freq_base > 0.0f) ? hdr.rope_freq_base : 10000.0f;
    ctx->info.bos_id      = tok_bos_id(ctx->tok);
    ctx->info.eos_id      = tok_eos_id(ctx->tok);
    ctx->info.add_bos_default = 1;
    ctx->info.is_chatml   = 0;
    ctx->rope_neox        = (hdr.flags & 2u) ? 1 : 0;

    uint32_t V = ctx->info.vocab_size;
    uint32_t D = ctx->info.dim;
    uint32_t H = ctx->info.hidden_dim;
    uint32_t L = ctx->info.n_layers;

    const uint8_t *p = ctx->mapped_ptr + sizeof(Idxt2Header);
    const uint8_t *end = ctx->mapped_ptr + ctx->mapped_size;

    ctx->w.tok_emb = (TensorView){ p, qt, V, D };
    p += (size_t)V * tensor_row_bytes(qt, D);

    ctx->w.layers = calloc(L, sizeof(*ctx->w.layers));
    for (uint32_t i = 0; i < L; i++) {
        if (p >= end) { snprintf(err_buf, err_len, "Truncated IDX2 file"); return -1; }
        TensorView an = { p, qt, 1, D };
        p += tensor_row_bytes(qt, D);
        ctx->w.layers[i].attn_norm = unpack_1d_tensor(&an);

        ctx->w.layers[i].wq = (TensorView){ p, qt, Q_D, D };   p += (size_t)Q_D  * tensor_row_bytes(qt, D);
        ctx->w.layers[i].wk = (TensorView){ p, qt, KV_D, D };  p += (size_t)KV_D * tensor_row_bytes(qt, D);
        ctx->w.layers[i].wv = (TensorView){ p, qt, KV_D, D };  p += (size_t)KV_D * tensor_row_bytes(qt, D);
        ctx->w.layers[i].wo = (TensorView){ p, qt, D, Q_D };   p += (size_t)D    * tensor_row_bytes(qt, Q_D);

        TensorView fn = { p, qt, 1, D };
        p += tensor_row_bytes(qt, D);
        ctx->w.layers[i].ffn_norm = unpack_1d_tensor(&fn);

        ctx->w.layers[i].w1 = (TensorView){ p, qt, H, D }; p += (size_t)H * tensor_row_bytes(qt, D);
        ctx->w.layers[i].w2 = (TensorView){ p, qt, D, H }; p += (size_t)D * tensor_row_bytes(qt, H);
        ctx->w.layers[i].w3 = (TensorView){ p, qt, H, D }; p += (size_t)H * tensor_row_bytes(qt, D);
        ctx->w.layers[i].has_w1 = 1;
    }
    ctx->emb_scale = 1.0f;

    TensorView nf = { p, qt, 1, D };
    p += tensor_row_bytes(qt, D);
    if (p > end) {
        snprintf(err_buf, err_len, "Truncated IDX2 file");
        return -1;
    }
    ctx->w.norm_final = unpack_1d_tensor(&nf);

    if (tied) {
        ctx->w.lm_head = ctx->w.tok_emb;
    } else {
        ctx->w.lm_head = (TensorView){ p, qt, V, D };
        p += (size_t)V * tensor_row_bytes(qt, D);
        if (p > end) {
            snprintf(err_buf, err_len, "Truncated IDX2 lm_head");
            return -1;
        }
    }
    return 0;
}

static int get_gguf_tv(const GgufFile *gf, const uint8_t *base, const char *name, TensorView *out, int req, char *err, size_t elen) {
    const GgufTensorInfo *ti = gguf_find_tensor(gf, name);
    if (!ti) {
        if (req) snprintf(err, elen, "Missing GGUF tensor: %s", name);
        return -1;
    }
    out->data = base + gf->data_offset + ti->offset;
    out->type = ti->type;
    out->cols = (uint32_t)ti->dims[0];
    out->rows = (ti->n_dims >= 2) ? (uint32_t)ti->dims[1] : 1;
    return 0;
}

static int load_gguf(IndextContext *ctx, const char *tok_path, char *err_buf, size_t err_len) {
    ctx->gguf = gguf_parse_memory(ctx->mapped_ptr, ctx->mapped_size, err_buf, err_len);
    if (!ctx->gguf) return -1;

    GgufFile *gf = ctx->gguf;
    if (tok_path && tok_path[0]) {
        ctx->tok = tok_load_json(tok_path, err_buf, err_len);
        if (!ctx->tok) return -1;
    } else if (gf->tokenizer) {
        ctx->tok = gf->tokenizer;
        gf->tokenizer = NULL;
    } else {
        snprintf(err_buf, err_len, "GGUF has no embedded tokenizer");
        return -1;
    }

    if (gf->n_heads == 0 || gf->dim == 0 || gf->n_layers == 0 || gf->vocab_size == 0) {
        snprintf(err_buf, err_len, "Invalid GGUF model dimensions");
        return -1;
    }

    ctx->info.format = INDEXT_FMT_GGUF;
    snprintf(ctx->info.arch, sizeof(ctx->info.arch), "%s", gf->arch[0] ? gf->arch : "llama");
    snprintf(ctx->info.model_name, sizeof(ctx->info.model_name), "%s", gf->name[0] ? gf->name : "GGUF Model");
    ctx->info.vocab_size     = gf->vocab_size;
    ctx->info.dim            = gf->dim;
    ctx->info.n_layers       = gf->n_layers;
    ctx->info.n_heads        = gf->n_heads;
    ctx->info.n_kv_heads     = gf->n_kv_heads ? gf->n_kv_heads : gf->n_heads;
    ctx->info.hidden_dim     = gf->hidden_dim;
    ctx->info.max_seq_len    = gf->max_seq_len ? gf->max_seq_len : 2048;
    ctx->info.head_dim       = gf->head_dim ? gf->head_dim : (gf->dim / gf->n_heads);
    ctx->rope_dim            = gf->rope_dim ? gf->rope_dim : ctx->info.head_dim;
    ctx->info.rms_eps        = gf->rms_eps;
    ctx->info.rope_freq_base = gf->rope_freq_base;
    ctx->info.bos_id         = gf->bos_id;
    ctx->info.eos_id         = gf->eos_id;
    ctx->info.add_bos_default = (gf->add_bos == 0) ? 0 : 1;
    ctx->info.is_chatml      = gf->is_chatml;
    ctx->rope_neox = (
        strcmp(ctx->info.arch, "qwen2") == 0 ||
        strcmp(ctx->info.arch, "qwen3") == 0 ||
        strcmp(ctx->info.arch, "gemma") == 0 ||
        strcmp(ctx->info.arch, "gemma2") == 0 ||
        strcmp(ctx->info.arch, "gemma3") == 0 ||
        strcmp(ctx->info.arch, "phi3") == 0 ||
        strcmp(ctx->info.arch, "phi2") == 0 ||
        strcmp(ctx->info.arch, "stablelm") == 0 ||
        strcmp(ctx->info.arch, "starcoder2") == 0 ||
        strcmp(ctx->info.arch, "gptneox") == 0
    ) ? 1 : 0;
    ctx->use_gelu = (
        strcmp(ctx->info.arch, "gemma") == 0 ||
        strcmp(ctx->info.arch, "gemma2") == 0 ||
        strcmp(ctx->info.arch, "gemma3") == 0 ||
        strcmp(ctx->info.arch, "gpt2") == 0 ||
        strcmp(ctx->info.arch, "starcoder2") == 0 ||
        strcmp(ctx->info.arch, "phi2") == 0
    ) ? 1 : 0;
    ctx->emb_scale = (
        strcmp(ctx->info.arch, "gemma") == 0 ||
        strcmp(ctx->info.arch, "gemma2") == 0 ||
        strcmp(ctx->info.arch, "gemma3") == 0
    ) ? sqrtf((float)ctx->info.dim) : 1.0f;
    ctx->attn_softcap = gf->attn_logit_softcapping;
    ctx->final_softcap = gf->final_logit_softcapping;

    if (get_gguf_tv(gf, ctx->mapped_ptr, "token_embd.weight", &ctx->w.tok_emb, 1, err_buf, err_len) != 0) return -1;
    if (get_gguf_tv(gf, ctx->mapped_ptr, "output.weight", &ctx->w.lm_head, 0, err_buf, err_len) != 0) {
        ctx->w.lm_head = ctx->w.tok_emb;
    }

    TensorView nf;
    if (get_gguf_tv(gf, ctx->mapped_ptr, "output_norm.weight", &nf, 1, err_buf, err_len) != 0) return -1;
    ctx->w.norm_final = unpack_1d_tensor(&nf);
    if (get_gguf_tv(gf, ctx->mapped_ptr, "output_norm.bias", &nf, 0, err_buf, err_len) == 0) {
        ctx->w.norm_final_b = unpack_1d_tensor(&nf);
    }

    ctx->w.layers = calloc(ctx->info.n_layers, sizeof(*ctx->w.layers));
    char tname[128];
    for (uint32_t i = 0; i < ctx->info.n_layers; i++) {
        TensorView tv;
        snprintf(tname, sizeof(tname), "blk.%u.attn_norm.weight", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &tv, 1, err_buf, err_len) != 0) return -1;
        ctx->w.layers[i].attn_norm = unpack_1d_tensor(&tv);
        snprintf(tname, sizeof(tname), "blk.%u.attn_norm.bias", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &tv, 0, err_buf, err_len) == 0) ctx->w.layers[i].attn_norm_b = unpack_1d_tensor(&tv);

        snprintf(tname, sizeof(tname), "blk.%u.post_attention_norm.weight", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &tv, 0, err_buf, err_len) == 0) ctx->w.layers[i].post_attn_norm = unpack_1d_tensor(&tv);

        snprintf(tname, sizeof(tname), "blk.%u.ffn_norm.weight", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &tv, 1, err_buf, err_len) != 0) return -1;
        ctx->w.layers[i].ffn_norm = unpack_1d_tensor(&tv);
        snprintf(tname, sizeof(tname), "blk.%u.ffn_norm.bias", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &tv, 0, err_buf, err_len) == 0) ctx->w.layers[i].ffn_norm_b = unpack_1d_tensor(&tv);

        snprintf(tname, sizeof(tname), "blk.%u.post_ffw_norm.weight", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &tv, 0, err_buf, err_len) == 0) ctx->w.layers[i].post_ffn_norm = unpack_1d_tensor(&tv);

        snprintf(tname, sizeof(tname), "blk.%u.attn_q_norm.weight", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &tv, 0, err_buf, err_len) == 0) ctx->w.layers[i].q_norm = unpack_1d_tensor(&tv);
        snprintf(tname, sizeof(tname), "blk.%u.attn_k_norm.weight", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &tv, 0, err_buf, err_len) == 0) ctx->w.layers[i].k_norm = unpack_1d_tensor(&tv);

        snprintf(tname, sizeof(tname), "blk.%u.attn_q.weight", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &ctx->w.layers[i].wq, 0, err_buf, err_len) == 0) {
            snprintf(tname, sizeof(tname), "blk.%u.attn_k.weight", i);
            if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &ctx->w.layers[i].wk, 1, err_buf, err_len) != 0) return -1;
            snprintf(tname, sizeof(tname), "blk.%u.attn_v.weight", i);
            if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &ctx->w.layers[i].wv, 1, err_buf, err_len) != 0) return -1;
        } else {
            snprintf(tname, sizeof(tname), "blk.%u.attn_qkv.weight", i);
            if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &ctx->w.layers[i].wqkv, 1, err_buf, err_len) != 0) return -1;
            ctx->w.layers[i].has_wqkv = 1;
        }
        snprintf(tname, sizeof(tname), "blk.%u.attn_output.weight", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &ctx->w.layers[i].wo, 1, err_buf, err_len) != 0) return -1;

        snprintf(tname, sizeof(tname), "blk.%u.attn_q.bias", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &tv, 0, err_buf, err_len) == 0) ctx->w.layers[i].bq = unpack_1d_tensor(&tv);
        snprintf(tname, sizeof(tname), "blk.%u.attn_k.bias", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &tv, 0, err_buf, err_len) == 0) ctx->w.layers[i].bk = unpack_1d_tensor(&tv);
        snprintf(tname, sizeof(tname), "blk.%u.attn_v.bias", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &tv, 0, err_buf, err_len) == 0) ctx->w.layers[i].bv = unpack_1d_tensor(&tv);
        snprintf(tname, sizeof(tname), "blk.%u.attn_output.bias", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &tv, 0, err_buf, err_len) == 0) ctx->w.layers[i].bo = unpack_1d_tensor(&tv);

        snprintf(tname, sizeof(tname), "blk.%u.ffn_gate.weight", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &ctx->w.layers[i].w1, 0, err_buf, err_len) == 0) {
            ctx->w.layers[i].has_w1 = 1;
        }
        snprintf(tname, sizeof(tname), "blk.%u.ffn_down.weight", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &ctx->w.layers[i].w2, 1, err_buf, err_len) != 0) return -1;
        snprintf(tname, sizeof(tname), "blk.%u.ffn_up.weight", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &ctx->w.layers[i].w3, 1, err_buf, err_len) != 0) return -1;

        snprintf(tname, sizeof(tname), "blk.%u.ffn_gate.bias", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &tv, 0, err_buf, err_len) == 0) ctx->w.layers[i].b1 = unpack_1d_tensor(&tv);
        snprintf(tname, sizeof(tname), "blk.%u.ffn_down.bias", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &tv, 0, err_buf, err_len) == 0) ctx->w.layers[i].b2 = unpack_1d_tensor(&tv);
        snprintf(tname, sizeof(tname), "blk.%u.ffn_up.bias", i);
        if (get_gguf_tv(gf, ctx->mapped_ptr, tname, &tv, 0, err_buf, err_len) == 0) ctx->w.layers[i].b3 = unpack_1d_tensor(&tv);
    }
    ctx->info.quant_type = ctx->w.layers[0].has_wqkv ? ctx->w.layers[0].wqkv.type : ctx->w.layers[0].wq.type;
    return 0;
}

IndextParams indext_default_params(void) {
    IndextParams p;
    p.n_threads = 0;
    p.n_ctx = 0;
    p.max_tokens = 128;
    p.temperature = 0.7f;
    p.top_p = 0.9f;
    p.top_k = 40;
    p.repeat_penalty = 1.1f;
    p.seed = 0;
    p.use_mmap = 1;
    p.kv_mode = 2;
    return p;
}

IndextContext *indext_load(const char *model_path, const char *tokenizer_path, const IndextParams *params, char *err_buf, size_t err_len) {
    double t0 = now_ms();
    init_fp16_lut();
    IndextContext *ctx = (IndextContext *)calloc(1, sizeof(IndextContext));
#ifndef _WIN32
    ctx->fd = -1;
#else
    ctx->hFile = INVALID_HANDLE_VALUE;
#endif
    ctx->params = params ? *params : indext_default_params();
    if (ctx->params.n_threads <= 0) ctx->params.n_threads = detect_cpu_cores();
    ctx->rng_state = ctx->params.seed ? (uint64_t)ctx->params.seed : (uint64_t)time(NULL) ^ 0x9E3779B97F4A7C15ULL;

    int frc = ctx->params.use_mmap
        ? map_file_ro(ctx, model_path, err_buf, err_len)
        : load_file_ram(ctx, model_path, err_buf, err_len);
    if (frc != 0) {
        indext_free(ctx);
        return NULL;
    }
    ctx->info.file_size = ctx->mapped_size;
    ctx->info.used_mmap = ctx->used_ram_buf ? 0 : 1;
    int km = ctx->params.kv_mode;
    if (km < 0 || km > 3) km = 2;
    ctx->info.kv_mode = km;

    if (ctx->mapped_size < 4) {
        snprintf(err_buf, err_len, "File too small");
        indext_free(ctx);
        return NULL;
    }

    int rc = -1;
    if (memcmp(ctx->mapped_ptr, "IDX2", 4) == 0) {
        rc = load_idxt2(ctx, tokenizer_path, err_buf, err_len);
    } else if (memcmp(ctx->mapped_ptr, "IDXT", 4) == 0) {
        rc = load_idxt(ctx, tokenizer_path, err_buf, err_len);
    } else if (memcmp(ctx->mapped_ptr, "GGUF", 4) == 0) {
        rc = load_gguf(ctx, tokenizer_path, err_buf, err_len);
    } else {
        snprintf(err_buf, err_len, "Unknown model format (expected IDXT, IDX2, or GGUF)");
    }

    if (rc != 0) {
        indext_free(ctx);
        return NULL;
    }

    if (ctx->params.n_ctx > 0 && (uint32_t)ctx->params.n_ctx < ctx->info.max_seq_len) {
        ctx->info.max_seq_len = (uint32_t)ctx->params.n_ctx;
    } else if (ctx->params.n_ctx == 0) {
        uint32_t kv_d = ctx->info.n_kv_heads * ctx->info.head_dim;
        size_t bytes_per_tok = (km == 3)
            ? ((size_t)ctx->info.n_layers * (kv_d + ctx->info.n_kv_heads * 2 * sizeof(float)))
            : ((km == 2)
                ? ((size_t)ctx->info.n_layers * (kv_d * 2 * sizeof(int8_t) + ctx->info.n_kv_heads * 2 * sizeof(float)))
                : ((size_t)ctx->info.n_layers * kv_d * 2 * (km == 1 ? sizeof(uint16_t) : sizeof(float))));
        size_t max_kv_bytes = 96u * 1024u * 1024u;
        if (bytes_per_tok > 0) {
            uint32_t safe_s = (uint32_t)(max_kv_bytes / bytes_per_tok);
            if (safe_s < 128) safe_s = 128;
            if (safe_s > 2048) safe_s = 2048;
            if (ctx->info.max_seq_len > safe_s) ctx->info.max_seq_len = safe_s;
        }
    }

    uint32_t D = ctx->info.dim;
    uint32_t H = ctx->info.hidden_dim;
    uint32_t Q_D = ctx->info.n_heads * ctx->info.head_dim;
    uint32_t KV_D = ctx->info.n_kv_heads * ctx->info.head_dim;
    uint32_t L = ctx->info.n_layers;
    uint32_t S = ctx->info.max_seq_len;
    uint32_t V = ctx->info.vocab_size;
    if (ctx->rope_dim == 0) ctx->rope_dim = ctx->info.head_dim;
    uint32_t half_rope = ctx->rope_dim / 2;

    uint32_t max_x = Q_D + 2 * KV_D;
    if (D > max_x) max_x = D;
    ctx->x      = (float *)calloc(D, sizeof(float));
    ctx->xb     = (float *)calloc(max_x, sizeof(float));
    ctx->xb2    = (float *)calloc(D, sizeof(float));
    ctx->hb     = (float *)calloc(H * 2, sizeof(float));
    ctx->hb2    = (float *)calloc(H, sizeof(float));
    ctx->q      = (float *)calloc(Q_D, sizeof(float));
    ctx->k      = (float *)calloc(KV_D, sizeof(float));
    ctx->v      = (float *)calloc(KV_D, sizeof(float));
    ctx->att    = (float *)calloc((size_t)ctx->info.n_heads * S, sizeof(float));
    ctx->logits = (float *)calloc(V > D ? V : D, sizeof(float));

    size_t kv_elems = (size_t)L * S * KV_D;
    size_t kv_scale_elems = (size_t)L * S * ctx->info.n_kv_heads;
    int kv_ok = 0;
    if (km == 3) {
        size_t kv_bytes_q4 = kv_elems / 2;
        ctx->info.kv_cache_bytes = kv_bytes_q4 * 2 + kv_scale_elems * 2 * sizeof(float);
        ctx->key_cache_q4 = (uint8_t *)calloc(kv_bytes_q4, 1);
        ctx->val_cache_q4 = (uint8_t *)calloc(kv_bytes_q4, 1);
        ctx->key_scale_q4 = (float *)calloc(kv_scale_elems, sizeof(float));
        ctx->val_scale_q4 = (float *)calloc(kv_scale_elems, sizeof(float));
        kv_ok = (ctx->key_cache_q4 && ctx->val_cache_q4 && ctx->key_scale_q4 && ctx->val_scale_q4);
    } else if (km == 2) {
        ctx->info.kv_cache_bytes = kv_elems * 2 * sizeof(int8_t) + kv_scale_elems * 2 * sizeof(float);
        ctx->key_cache_q8 = (int8_t *)calloc(kv_elems, sizeof(int8_t));
        ctx->val_cache_q8 = (int8_t *)calloc(kv_elems, sizeof(int8_t));
        ctx->key_scale_q8 = (float *)calloc(kv_scale_elems, sizeof(float));
        ctx->val_scale_q8 = (float *)calloc(kv_scale_elems, sizeof(float));
        kv_ok = (ctx->key_cache_q8 && ctx->val_cache_q8 && ctx->key_scale_q8 && ctx->val_scale_q8);
    } else if (km == 1) {
        ctx->info.kv_cache_bytes = kv_elems * 2 * sizeof(uint16_t);
        ctx->key_cache_f16 = (uint16_t *)calloc(kv_elems, sizeof(uint16_t));
        ctx->val_cache_f16 = (uint16_t *)calloc(kv_elems, sizeof(uint16_t));
        kv_ok = (ctx->key_cache_f16 && ctx->val_cache_f16);
    } else {
        ctx->info.kv_cache_bytes = kv_elems * 2 * sizeof(float);
        ctx->key_cache = (float *)calloc(kv_elems, sizeof(float));
        ctx->val_cache = (float *)calloc(kv_elems, sizeof(float));
        kv_ok = (ctx->key_cache && ctx->val_cache);
    }
    ctx->recent_tokens = (int *)calloc(S, sizeof(int));
    uint32_t max_cols = (H > max_x ? H : max_x);
    ctx->qx_buf = (BlockQ8 *)calloc((max_cols / 32) + 4, sizeof(BlockQ8));
    ctx->rope_inv_freq = (float *)calloc(half_rope + 1, sizeof(float));
    ctx->rope_cos      = (float *)calloc(half_rope + 1, sizeof(float));
    ctx->rope_sin      = (float *)calloc(half_rope + 1, sizeof(float));
    ctx->probs_buf     = (ProbIdx *)calloc(V, sizeof(ProbIdx));

    if (!ctx->x || !ctx->xb || !ctx->xb2 || !ctx->hb || !ctx->hb2 ||
        !ctx->q || !ctx->k || !ctx->v || !ctx->att || !ctx->logits ||
        !kv_ok || !ctx->recent_tokens || !ctx->qx_buf ||
        !ctx->rope_inv_freq || !ctx->rope_cos || !ctx->rope_sin || !ctx->probs_buf) {
        snprintf(err_buf, err_len, "Out of memory allocating KV cache (%.1f MB). Lower Context Length.",
                 (double)ctx->info.kv_cache_bytes / (1024.0 * 1024.0));
        indext_free(ctx);
        return NULL;
    }

    for (uint32_t i = 0; i < half_rope; i++) {
        ctx->rope_inv_freq[i] = 1.0f / powf(ctx->info.rope_freq_base, (float)(2 * i) / (float)ctx->rope_dim);
    }

#ifndef _WIN32
    pool_init(ctx, ctx->params.n_threads);
#endif

    ctx->stats.load_time_ms = now_ms() - t0;
    return ctx;
}

void indext_free(IndextContext *ctx) {
    if (!ctx) return;
#ifndef _WIN32
    pool_destroy(ctx);
#endif
    if (ctx->w.layers) {
        for (uint32_t i = 0; i < ctx->info.n_layers; i++) {
            free(ctx->w.layers[i].attn_norm);
            free(ctx->w.layers[i].attn_norm_b);
            free(ctx->w.layers[i].post_attn_norm);
            free(ctx->w.layers[i].ffn_norm);
            free(ctx->w.layers[i].ffn_norm_b);
            free(ctx->w.layers[i].post_ffn_norm);
            free(ctx->w.layers[i].q_norm);
            free(ctx->w.layers[i].k_norm);
            free(ctx->w.layers[i].bq);
            free(ctx->w.layers[i].bk);
            free(ctx->w.layers[i].bv);
            free(ctx->w.layers[i].bo);
            free(ctx->w.layers[i].b1);
            free(ctx->w.layers[i].b2);
            free(ctx->w.layers[i].b3);
        }
        free(ctx->w.layers);
    }
    free(ctx->w.norm_final);
    free(ctx->w.norm_final_b);
    free(ctx->x); free(ctx->xb); free(ctx->xb2);
    free(ctx->hb); free(ctx->hb2);
    free(ctx->q); free(ctx->k); free(ctx->v);
    free(ctx->att); free(ctx->logits);
    free(ctx->key_cache); free(ctx->val_cache);
    free(ctx->key_cache_f16); free(ctx->val_cache_f16);
    free(ctx->key_cache_q8); free(ctx->val_cache_q8);
    free(ctx->key_scale_q8); free(ctx->val_scale_q8);
    free(ctx->key_cache_q4); free(ctx->val_cache_q4);
    free(ctx->key_scale_q4); free(ctx->val_scale_q4);
    free(ctx->recent_tokens);
    free(ctx->qx_buf);
    free(ctx->rope_inv_freq); free(ctx->rope_cos); free(ctx->rope_sin);
    free(ctx->probs_buf);
    tok_free(ctx->tok);
    gguf_free(ctx->gguf);
    unmap_file_ro(ctx);
    free(ctx);
}

void indext_reset_kv(IndextContext *ctx) {
    if (!ctx) return;
    ctx->cur_pos = 0;
    ctx->n_recent = 0;
}

void indext_get_info(const IndextContext *ctx, IndextModelInfo *out_info) {
    if (ctx && out_info) *out_info = ctx->info;
}

void indext_get_stats(const IndextContext *ctx, IndextStats *out_stats) {
    if (ctx && out_stats) *out_stats = ctx->stats;
}

void indext_update_params(IndextContext *ctx, const IndextParams *params) {
    if (!ctx || !params) return;
    int old_threads = ctx->params.n_threads;
    int old_mmap = ctx->params.use_mmap;
    int old_kv_mode = ctx->params.kv_mode;
    ctx->params = *params;
    if (ctx->params.n_threads <= 0) ctx->params.n_threads = old_threads;
    ctx->params.use_mmap = old_mmap;
    ctx->params.kv_mode = old_kv_mode;
    if (ctx->params.seed != 0) ctx->rng_state = (uint64_t)ctx->params.seed;
}

void indext_request_stop(IndextContext *ctx) {
    if (ctx) ctx->stop_requested = 1;
}

static float *forward_token(IndextContext *ctx, int token, int pos, int compute_logits) {
    uint32_t D = ctx->info.dim;
    uint32_t H = ctx->info.hidden_dim;
    uint32_t L = ctx->info.n_layers;
    uint32_t S = ctx->info.max_seq_len;
    uint32_t n_heads = ctx->info.n_heads;
    uint32_t n_kv = ctx->info.n_kv_heads;
    uint32_t hd = ctx->info.head_dim;
    uint32_t Q_D = n_heads * hd;
    uint32_t kv_dim = n_kv * hd;
    uint32_t kv_mul = n_heads / n_kv;
    int nt = ctx->params.n_threads;
    int kv_mode = ctx->info.kv_mode;

    uint32_t half_rope = ctx->rope_dim / 2;
    for (uint32_t i = 0; i < half_rope; i++) {
        float val = (float)pos * ctx->rope_inv_freq[i];
        ctx->rope_cos[i] = cosf(val);
        ctx->rope_sin[i] = sinf(val);
    }

    if (token < 0 || (uint32_t)token >= ctx->info.vocab_size) token = 0;
    dequant_row(&ctx->w.tok_emb, (uint32_t)token, ctx->x);
    if (ctx->emb_scale != 1.0f) {
        for (uint32_t i = 0; i < D; i++) ctx->x[i] *= ctx->emb_scale;
    }

    for (uint32_t l = 0; l < L; l++) {
        if (ctx->w.layers[l].attn_norm_b) {
            layernorm(ctx->xb, ctx->x, ctx->w.layers[l].attn_norm, ctx->w.layers[l].attn_norm_b, D, ctx->info.rms_eps);
        } else {
            rmsnorm(ctx->xb, ctx->x, ctx->w.layers[l].attn_norm, D, ctx->info.rms_eps);
        }

        ctx->qx_valid = 0;
        if (ctx->w.layers[l].has_wqkv) {
            /* Fused QKV (Phi-3 style): xb holds [Q_D | KV_D | KV_D] temporarily? Use xb2? No, quantize xb first into qx_buf, then write into xb! */
            if (needs_q8_x(ctx->w.layers[l].wqkv.type)) {
                quantize_x_q8(ctx->xb, ctx->qx_buf, D);
                ctx->qx_valid = 1;
            }
            /* Note: if F32/F16/BF16, copy xb to probs_buf? Better: only overwrite xb if Q8; for non-Q8, copy x? Let's use a stack/temp or copy xb->logits[0..D] since logits is unused until the end! */
            memcpy(ctx->logits, ctx->xb, (size_t)D * sizeof(float));
            matmul_mt_ex(ctx, &ctx->w.layers[l].wqkv, ctx->logits, ctx->xb, nt, 0);
            memcpy(ctx->q, ctx->xb, (size_t)Q_D * sizeof(float));
            memcpy(ctx->k, ctx->xb + Q_D, (size_t)kv_dim * sizeof(float));
            memcpy(ctx->v, ctx->xb + Q_D + kv_dim, (size_t)kv_dim * sizeof(float));
        } else {
            matmul_mt_ex(ctx, &ctx->w.layers[l].wq, ctx->xb, ctx->q, nt, 0);
            matmul_mt_ex(ctx, &ctx->w.layers[l].wk, ctx->xb, ctx->k, nt, 0);
            matmul_mt_ex(ctx, &ctx->w.layers[l].wv, ctx->xb, ctx->v, nt, 0);
        }
        if (ctx->w.layers[l].bq) vec_add_f32(ctx->q, ctx->w.layers[l].bq, Q_D);
        if (ctx->w.layers[l].bk) vec_add_f32(ctx->k, ctx->w.layers[l].bk, kv_dim);
        if (ctx->w.layers[l].bv) vec_add_f32(ctx->v, ctx->w.layers[l].bv, kv_dim);

        if (ctx->w.layers[l].q_norm) {
            for (uint32_t h = 0; h < n_heads; h++) rmsnorm(ctx->q + h * hd, ctx->q + h * hd, ctx->w.layers[l].q_norm, hd, ctx->info.rms_eps);
        }
        if (ctx->w.layers[l].k_norm) {
            for (uint32_t h = 0; h < n_kv; h++) rmsnorm(ctx->k + h * hd, ctx->k + h * hd, ctx->w.layers[l].k_norm, hd, ctx->info.rms_eps);
        }

        apply_rope_cached(ctx->q, n_heads, hd, ctx->rope_dim, ctx->rope_cos, ctx->rope_sin, ctx->rope_neox);
        apply_rope_cached(ctx->k, n_kv, hd, ctx->rope_dim, ctx->rope_cos, ctx->rope_sin, ctx->rope_neox);

        size_t loff = (size_t)l * S * kv_dim;
        size_t soff = (size_t)l * S * n_kv;
        if (kv_mode == 3) {
            uint32_t half_hd = hd / 2;
            uint8_t *kd = ctx->key_cache_q4 + (loff / 2) + (size_t)pos * (kv_dim / 2);
            uint8_t *vd = ctx->val_cache_q4 + (loff / 2) + (size_t)pos * (kv_dim / 2);
            float *ks = ctx->key_scale_q4 + soff + (size_t)pos * n_kv;
            float *vs = ctx->val_scale_q4 + soff + (size_t)pos * n_kv;
            for (uint32_t kvh = 0; kvh < n_kv; kvh++) {
                const float *ksrc = ctx->k + kvh * hd;
                const float *vsrc = ctx->v + kvh * hd;
                uint8_t *kdst = kd + kvh * half_hd;
                uint8_t *vdst = vd + kvh * half_hd;
                float kamax = 0.0f, vamax = 0.0f;
                for (uint32_t i = 0; i < hd; i++) {
                    float ak = ksrc[i] < 0 ? -ksrc[i] : ksrc[i];
                    float av = vsrc[i] < 0 ? -vsrc[i] : vsrc[i];
                    if (ak > kamax) kamax = ak;
                    if (av > vamax) vamax = av;
                }
                float ksc = kamax / 7.0f;
                float vsc = vamax / 7.0f;
                float ik = ksc > 0.0f ? (7.0f / kamax) : 0.0f;
                float iv = vsc > 0.0f ? (7.0f / vamax) : 0.0f;
                ks[kvh] = ksc;
                vs[kvh] = vsc;
                for (uint32_t j = 0; j < half_hd; j++) {
                    float fk0 = ksrc[j] * ik, fk1 = ksrc[j + half_hd] * ik;
                    int qk0 = (int)(fk0 + (fk0 >= 0 ? 0.5f : -0.5f)) + 8;
                    int qk1 = (int)(fk1 + (fk1 >= 0 ? 0.5f : -0.5f)) + 8;
                    if (qk0 < 0) qk0 = 0; else if (qk0 > 15) qk0 = 15;
                    if (qk1 < 0) qk1 = 0; else if (qk1 > 15) qk1 = 15;
                    kdst[j] = (uint8_t)(qk0 | (qk1 << 4));

                    float fv0 = vsrc[j] * iv, fv1 = vsrc[j + half_hd] * iv;
                    int qv0 = (int)(fv0 + (fv0 >= 0 ? 0.5f : -0.5f)) + 8;
                    int qv1 = (int)(fv1 + (fv1 >= 0 ? 0.5f : -0.5f)) + 8;
                    if (qv0 < 0) qv0 = 0; else if (qv0 > 15) qv0 = 15;
                    if (qv1 < 0) qv1 = 0; else if (qv1 > 15) qv1 = 15;
                    vdst[j] = (uint8_t)(qv0 | (qv1 << 4));
                }
            }
        } else if (kv_mode == 2) {
            int8_t *kd = ctx->key_cache_q8 + loff + (size_t)pos * kv_dim;
            int8_t *vd = ctx->val_cache_q8 + loff + (size_t)pos * kv_dim;
            float *ks = ctx->key_scale_q8 + soff + (size_t)pos * n_kv;
            float *vs = ctx->val_scale_q8 + soff + (size_t)pos * n_kv;
            for (uint32_t kvh = 0; kvh < n_kv; kvh++) {
                const float *ksrc = ctx->k + kvh * hd;
                const float *vsrc = ctx->v + kvh * hd;
                int8_t *kdst = kd + kvh * hd;
                int8_t *vdst = vd + kvh * hd;
                float kamax = 0.0f, vamax = 0.0f;
                for (uint32_t i = 0; i < hd; i++) {
                    float ak = ksrc[i] < 0 ? -ksrc[i] : ksrc[i];
                    float av = vsrc[i] < 0 ? -vsrc[i] : vsrc[i];
                    if (ak > kamax) kamax = ak;
                    if (av > vamax) vamax = av;
                }
                float ksc = kamax / 127.0f;
                float vsc = vamax / 127.0f;
                float ik = ksc > 0.0f ? (127.0f / kamax) : 0.0f;
                float iv = vsc > 0.0f ? (127.0f / vamax) : 0.0f;
                ks[kvh] = ksc;
                vs[kvh] = vsc;
                for (uint32_t i = 0; i < hd; i++) {
                    float fk = ksrc[i] * ik;
                    int qk = (int)(fk + (fk >= 0 ? 0.5f : -0.5f));
                    if (qk > 127) qk = 127; else if (qk < -127) qk = -127;
                    kdst[i] = (int8_t)qk;

                    float fv = vsrc[i] * iv;
                    int qv = (int)(fv + (fv >= 0 ? 0.5f : -0.5f));
                    if (qv > 127) qv = 127; else if (qv < -127) qv = -127;
                    vdst[i] = (int8_t)qv;
                }
            }
        } else if (kv_mode == 1) {
            uint16_t *kd = ctx->key_cache_f16 + loff + (size_t)pos * kv_dim;
            uint16_t *vd = ctx->val_cache_f16 + loff + (size_t)pos * kv_dim;
            for (uint32_t i = 0; i < kv_dim; i++) {
                kd[i] = fp32_to_fp16(ctx->k[i]);
                vd[i] = fp32_to_fp16(ctx->v[i]);
            }
        } else {
            memcpy(ctx->key_cache + loff + (size_t)pos * kv_dim, ctx->k, (size_t)kv_dim * sizeof(float));
            memcpy(ctx->val_cache + loff + (size_t)pos * kv_dim, ctx->v, (size_t)kv_dim * sizeof(float));
        }

        float scale = 1.0f / sqrtf((float)hd);
        for (uint32_t h = 0; h < n_heads; h++) {
            float *qh = ctx->q + h * hd;
            float *atth = ctx->att + (size_t)h * S;
            uint32_t kvh = h / kv_mul;

            if (kv_mode == 3) {
                uint32_t half_hd = hd / 2;
                if (hd <= 256 && (hd % 32) == 0) {
                    int8_t q_q8[256];
                    float qamax = 0.0f;
                    for (uint32_t i = 0; i < hd; i++) {
                        float aq = qh[i] < 0 ? -qh[i] : qh[i];
                        if (aq > qamax) qamax = aq;
                    }
                    float qsc = qamax / 127.0f;
                    float iq = qsc > 0.0f ? (127.0f / qamax) : 0.0f;
                    for (uint32_t i = 0; i < hd; i++) {
                        float fq = qh[i] * iq;
                        int qq = (int)(fq + (fq >= 0 ? 0.5f : -0.5f));
                        if (qq > 127) qq = 127; else if (qq < -127) qq = -127;
                        q_q8[i] = (int8_t)qq;
                    }
                    float qsc_scaled = qsc * scale;
#ifdef __ARM_NEON
                    uint8x16_t vmask4 = vdupq_n_u8(0x0F);
                    int8x16_t v8 = vdupq_n_s8(8);
#endif
                    for (int t = 0; t <= pos; t++) {
                        const uint8_t *kh = ctx->key_cache_q4 + (loff / 2) + (size_t)t * (kv_dim / 2) + kvh * half_hd;
                        float ksc = ctx->key_scale_q4[soff + (size_t)t * n_kv + kvh];
                        int32_t isum = 0;
#ifdef __ARM_NEON
                        int32x4_t acc = vdupq_n_s32(0);
                        for (uint32_t j = 0; j + 16 <= half_hd; j += 16) {
                            uint8x16_t vq = vld1q_u8(kh + j);
                            int8x16_t klo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vq, vmask4)), v8);
                            int8x16_t khi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vq, 4)), v8);
                            int8x16_t x0 = vld1q_s8(q_q8 + j);
                            int8x16_t x1 = vld1q_s8(q_q8 + j + half_hd);
                            acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(klo),  vget_low_s8(x0)));
                            acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(klo), vget_high_s8(x0)));
                            acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(khi),  vget_low_s8(x1)));
                            acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(khi), vget_high_s8(x1)));
                        }
                        int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
                        isum = vget_lane_s32(vpadd_s32(s2, s2), 0);
#else
                        for (uint32_t j = 0; j < half_hd; j++) {
                            isum += ((int)(kh[j] & 0x0F) - 8) * (int32_t)q_q8[j];
                            isum += ((int)(kh[j] >> 4)   - 8) * (int32_t)q_q8[j + half_hd];
                        }
#endif
                        atth[t] = (float)isum * qsc_scaled * ksc;
                    }
                } else {
                    for (int t = 0; t <= pos; t++) {
                        const uint8_t *kh = ctx->key_cache_q4 + (loff / 2) + (size_t)t * (kv_dim / 2) + kvh * half_hd;
                        float ksc = ctx->key_scale_q4[soff + (size_t)t * n_kv + kvh];
                        float score = 0.0f;
                        for (uint32_t j = 0; j < half_hd; j++) {
                            score += qh[j] * (float)((int)(kh[j] & 0x0F) - 8);
                            score += qh[j + half_hd] * (float)((int)(kh[j] >> 4) - 8);
                        }
                        atth[t] = score * ksc * scale;
                    }
                }
            } else if (kv_mode == 2) {
                if (hd <= 256 && (hd % 8) == 0) {
                    int8_t q_q8[256];
                    float qamax = 0.0f;
                    for (uint32_t i = 0; i < hd; i++) {
                        float aq = qh[i] < 0 ? -qh[i] : qh[i];
                        if (aq > qamax) qamax = aq;
                    }
                    float qsc = qamax / 127.0f;
                    float iq = qsc > 0.0f ? (127.0f / qamax) : 0.0f;
                    for (uint32_t i = 0; i < hd; i++) {
                        float fq = qh[i] * iq;
                        int qq = (int)(fq + (fq >= 0 ? 0.5f : -0.5f));
                        if (qq > 127) qq = 127; else if (qq < -127) qq = -127;
                        q_q8[i] = (int8_t)qq;
                    }
                    float qsc_scaled = qsc * scale;
                    for (int t = 0; t <= pos; t++) {
                        const int8_t *kh = ctx->key_cache_q8 + loff + (size_t)t * kv_dim + kvh * hd;
                        float ksc = ctx->key_scale_q8[soff + (size_t)t * n_kv + kvh];
                        int32_t isum = 0;
#ifdef __ARM_NEON
                        int32x4_t acc = vdupq_n_s32(0);
                        for (uint32_t i = 0; i + 8 <= hd; i += 8) {
                            acc = vpadalq_s16(acc, vmull_s8(vld1_s8(q_q8 + i), vld1_s8(kh + i)));
                        }
                        int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
                        isum = vget_lane_s32(vpadd_s32(s2, s2), 0);
#else
                        for (uint32_t i = 0; i < hd; i++) isum += (int32_t)q_q8[i] * (int32_t)kh[i];
#endif
                        atth[t] = (float)isum * qsc_scaled * ksc;
                    }
                } else {
                    for (int t = 0; t <= pos; t++) {
                        const int8_t *kh = ctx->key_cache_q8 + loff + (size_t)t * kv_dim + kvh * hd;
                        float ksc = ctx->key_scale_q8[soff + (size_t)t * n_kv + kvh];
                        float score = 0.0f;
                        for (uint32_t i = 0; i < hd; i++) score += qh[i] * (float)kh[i];
                        atth[t] = score * ksc * scale;
                    }
                }
            } else if (kv_mode == 1) {
                for (int t = 0; t <= pos; t++) {
                    const uint16_t *kh = ctx->key_cache_f16 + loff + (size_t)t * kv_dim + kvh * hd;
                    float score = 0.0f;
#ifdef __ARM_NEON
                    float32x4_t vs = vdupq_n_f32(0.0f);
                    uint32_t i = 0;
                    for (; i + 4 <= hd; i += 4) {
                        float k4[4] = { fp16_to_fp32(kh[i]), fp16_to_fp32(kh[i + 1]), fp16_to_fp32(kh[i + 2]), fp16_to_fp32(kh[i + 3]) };
                        vs = vmlaq_f32(vs, vld1q_f32(qh + i), vld1q_f32(k4));
                    }
                    float32x2_t v2 = vadd_f32(vget_low_f32(vs), vget_high_f32(vs));
                    score = vget_lane_f32(vpadd_f32(v2, v2), 0);
                    for (; i < hd; i++) score += qh[i] * fp16_to_fp32(kh[i]);
#else
                    for (uint32_t i = 0; i < hd; i++) score += qh[i] * fp16_to_fp32(kh[i]);
#endif
                    atth[t] = score * scale;
                }
            } else {
                for (int t = 0; t <= pos; t++) {
                    const float *kh = ctx->key_cache + loff + (size_t)t * kv_dim + kvh * hd;
                    float score = 0.0f;
#ifdef __ARM_NEON
                    float32x4_t vs = vdupq_n_f32(0.0f);
                    uint32_t i = 0;
                    for (; i + 4 <= hd; i += 4) vs = vmlaq_f32(vs, vld1q_f32(qh + i), vld1q_f32(kh + i));
                    float32x2_t v2 = vadd_f32(vget_low_f32(vs), vget_high_f32(vs));
                    score = vget_lane_f32(vpadd_f32(v2, v2), 0);
                    for (; i < hd; i++) score += qh[i] * kh[i];
#else
                    for (uint32_t i = 0; i < hd; i++) score += qh[i] * kh[i];
#endif
                    atth[t] = score * scale;
                }
            }
            if (ctx->attn_softcap > 0.0f) {
                float cap = ctx->attn_softcap, inv_cap = 1.0f / cap;
                for (int t = 0; t <= pos; t++) atth[t] = cap * tanhf(atth[t] * inv_cap);
            }
            softmax(atth, pos + 1);

            float *xbh = ctx->xb + h * hd;
            memset(xbh, 0, (size_t)hd * sizeof(float));
            if (kv_mode == 3) {
                uint32_t half_hd = hd / 2;
#ifdef __ARM_NEON
                uint8x16_t vmask4 = vdupq_n_u8(0x0F);
                int8x16_t v8 = vdupq_n_s8(8);
#endif
                for (int t = 0; t <= pos; t++) {
                    const uint8_t *vh = ctx->val_cache_q4 + (loff / 2) + (size_t)t * (kv_dim / 2) + kvh * half_hd;
                    float a = atth[t] * ctx->val_scale_q4[soff + (size_t)t * n_kv + kvh];
#ifdef __ARM_NEON
                    float32x4_t va = vdupq_n_f32(a);
                    uint32_t j = 0;
                    for (; j + 16 <= half_hd; j += 16) {
                        uint8x16_t vq = vld1q_u8(vh + j);
                        int8x16_t vlo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vq, vmask4)), v8);
                        int8x16_t vhi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vq, 4)), v8);
                        int16x8_t lo0 = vmovl_s8(vget_low_s8(vlo));
                        int16x8_t lo1 = vmovl_s8(vget_high_s8(vlo));
                        int16x8_t hi0 = vmovl_s8(vget_low_s8(vhi));
                        int16x8_t hi1 = vmovl_s8(vget_high_s8(vhi));
                        vst1q_f32(xbh + j,          vmlaq_f32(vld1q_f32(xbh + j),          va, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo0)))));
                        vst1q_f32(xbh + j + 4,      vmlaq_f32(vld1q_f32(xbh + j + 4),      va, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo0)))));
                        vst1q_f32(xbh + j + 8,      vmlaq_f32(vld1q_f32(xbh + j + 8),      va, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo1)))));
                        vst1q_f32(xbh + j + 12,     vmlaq_f32(vld1q_f32(xbh + j + 12),     va, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo1)))));
                        vst1q_f32(xbh + half_hd + j,      vmlaq_f32(vld1q_f32(xbh + half_hd + j),      va, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi0)))));
                        vst1q_f32(xbh + half_hd + j + 4,  vmlaq_f32(vld1q_f32(xbh + half_hd + j + 4),  va, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi0)))));
                        vst1q_f32(xbh + half_hd + j + 8,  vmlaq_f32(vld1q_f32(xbh + half_hd + j + 8),  va, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi1)))));
                        vst1q_f32(xbh + half_hd + j + 12, vmlaq_f32(vld1q_f32(xbh + half_hd + j + 12), va, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi1)))));
                    }
                    for (; j < half_hd; j++) {
                        xbh[j]           += a * (float)((int)(vh[j] & 0x0F) - 8);
                        xbh[j + half_hd] += a * (float)((int)(vh[j] >> 4)   - 8);
                    }
#else
                    for (uint32_t j = 0; j < half_hd; j++) {
                        xbh[j]           += a * (float)((int)(vh[j] & 0x0F) - 8);
                        xbh[j + half_hd] += a * (float)((int)(vh[j] >> 4)   - 8);
                    }
#endif
                }
            } else if (kv_mode == 2) {
                for (int t = 0; t <= pos; t++) {
                    const int8_t *vh = ctx->val_cache_q8 + loff + (size_t)t * kv_dim + kvh * hd;
                    float a = atth[t] * ctx->val_scale_q8[soff + (size_t)t * n_kv + kvh];
#ifdef __ARM_NEON
                    float32x4_t va = vdupq_n_f32(a);
                    uint32_t i = 0;
                    for (; i + 8 <= hd; i += 8) {
                        int16x8_t v16 = vmovl_s8(vld1_s8(vh + i));
                        float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16)));
                        float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16)));
                        vst1q_f32(xbh + i,     vmlaq_f32(vld1q_f32(xbh + i),     va, f0));
                        vst1q_f32(xbh + i + 4, vmlaq_f32(vld1q_f32(xbh + i + 4), va, f1));
                    }
                    for (; i < hd; i++) xbh[i] += a * (float)vh[i];
#else
                    for (uint32_t i = 0; i < hd; i++) xbh[i] += a * (float)vh[i];
#endif
                }
            } else if (kv_mode == 1) {
                for (int t = 0; t <= pos; t++) {
                    const uint16_t *vh = ctx->val_cache_f16 + loff + (size_t)t * kv_dim + kvh * hd;
                    float a = atth[t];
#ifdef __ARM_NEON
                    float32x4_t va = vdupq_n_f32(a);
                    uint32_t i = 0;
                    for (; i + 4 <= hd; i += 4) {
                        float v4[4] = { fp16_to_fp32(vh[i]), fp16_to_fp32(vh[i + 1]), fp16_to_fp32(vh[i + 2]), fp16_to_fp32(vh[i + 3]) };
                        float32x4_t cur = vld1q_f32(xbh + i);
                        vst1q_f32(xbh + i, vmlaq_f32(cur, va, vld1q_f32(v4)));
                    }
                    for (; i < hd; i++) xbh[i] += a * fp16_to_fp32(vh[i]);
#else
                    for (uint32_t i = 0; i < hd; i++) xbh[i] += a * fp16_to_fp32(vh[i]);
#endif
                }
            } else {
                for (int t = 0; t <= pos; t++) {
                    const float *vh = ctx->val_cache + loff + (size_t)t * kv_dim + kvh * hd;
                    float a = atth[t];
#ifdef __ARM_NEON
                    float32x4_t va = vdupq_n_f32(a);
                    uint32_t i = 0;
                    for (; i + 4 <= hd; i += 4) {
                        float32x4_t cur = vld1q_f32(xbh + i);
                        vst1q_f32(xbh + i, vmlaq_f32(cur, va, vld1q_f32(vh + i)));
                    }
                    for (; i < hd; i++) xbh[i] += a * vh[i];
#else
                    for (uint32_t i = 0; i < hd; i++) xbh[i] += a * vh[i];
#endif
                }
            }
        }

        matmul_mt_ex(ctx, &ctx->w.layers[l].wo, ctx->xb, ctx->xb2, nt, 1);
        if (ctx->w.layers[l].bo) vec_add_f32(ctx->xb2, ctx->w.layers[l].bo, D);
        if (ctx->w.layers[l].post_attn_norm) rmsnorm(ctx->xb2, ctx->xb2, ctx->w.layers[l].post_attn_norm, D, ctx->info.rms_eps);
        vec_add_f32(ctx->x, ctx->xb2, D);

        if (ctx->w.layers[l].ffn_norm_b) {
            layernorm(ctx->xb, ctx->x, ctx->w.layers[l].ffn_norm, ctx->w.layers[l].ffn_norm_b, D, ctx->info.rms_eps);
        } else {
            rmsnorm(ctx->xb, ctx->x, ctx->w.layers[l].ffn_norm, D, ctx->info.rms_eps);
        }
        ctx->qx_valid = 0;
        if (ctx->w.layers[l].has_w1) {
            matmul_mt_ex(ctx, &ctx->w.layers[l].w1, ctx->xb, ctx->hb, nt, 0);
            matmul_mt_ex(ctx, &ctx->w.layers[l].w3, ctx->xb, ctx->hb2, nt, 0);
            if (ctx->w.layers[l].b1) vec_add_f32(ctx->hb, ctx->w.layers[l].b1, H);
            if (ctx->w.layers[l].b3) vec_add_f32(ctx->hb2, ctx->w.layers[l].b3, H);
            if (ctx->use_gelu) {
                for (uint32_t i = 0; i < H; i++) ctx->hb[i] = gelu_f32(ctx->hb[i]) * ctx->hb2[i];
            } else {
                for (uint32_t i = 0; i < H; i++) {
                    float val = ctx->hb[i];
                    ctx->hb[i] = (val / (1.0f + fast_expf(-val))) * ctx->hb2[i];
                }
            }
        } else if (ctx->w.layers[l].w3.rows == 2 * H) {
            /* Fused gate+up (Phi-3 style) */
            matmul_mt_ex(ctx, &ctx->w.layers[l].w3, ctx->xb, ctx->hb, nt, 0);
            if (ctx->w.layers[l].b3) vec_add_f32(ctx->hb, ctx->w.layers[l].b3, 2 * H);
            for (uint32_t i = 0; i < H; i++) {
                float val = ctx->hb[i];
                float up = ctx->hb[i + H];
                ctx->hb[i] = (ctx->use_gelu ? gelu_f32(val) : (val / (1.0f + fast_expf(-val)))) * up;
            }
        } else {
            /* Standard 2-layer MLP (GPT-2 / StarCoder2 / Phi-2) */
            matmul_mt_ex(ctx, &ctx->w.layers[l].w3, ctx->xb, ctx->hb, nt, 0);
            if (ctx->w.layers[l].b3) vec_add_f32(ctx->hb, ctx->w.layers[l].b3, H);
            for (uint32_t i = 0; i < H; i++) ctx->hb[i] = gelu_f32(ctx->hb[i]);
        }
        matmul_mt_ex(ctx, &ctx->w.layers[l].w2, ctx->hb, ctx->xb, nt, 1);
        if (ctx->w.layers[l].b2) vec_add_f32(ctx->xb, ctx->w.layers[l].b2, D);
        if (ctx->w.layers[l].post_ffn_norm) rmsnorm(ctx->xb, ctx->xb, ctx->w.layers[l].post_ffn_norm, D, ctx->info.rms_eps);
        vec_add_f32(ctx->x, ctx->xb, D);
    }

    if (!compute_logits) return NULL;
    if (ctx->w.norm_final_b) {
        layernorm(ctx->x, ctx->x, ctx->w.norm_final, ctx->w.norm_final_b, D, ctx->info.rms_eps);
    } else {
        rmsnorm(ctx->x, ctx->x, ctx->w.norm_final, D, ctx->info.rms_eps);
    }
    matmul_mt_ex(ctx, &ctx->w.lm_head, ctx->x, ctx->logits, nt, 1);
    if (ctx->final_softcap > 0.0f) {
        float cap = ctx->final_softcap, inv_cap = 1.0f / cap;
        uint32_t V = ctx->info.vocab_size;
        for (uint32_t i = 0; i < V; i++) ctx->logits[i] = cap * tanhf(ctx->logits[i] * inv_cap);
    }
    return ctx->logits;
}

static int cmp_prob_desc(const void *a, const void *b) {
    float pa = ((const ProbIdx *)a)->prob;
    float pb = ((const ProbIdx *)b)->prob;
    return (pb > pa) - (pb < pa);
}

static float rand_f32(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (float)(((*state * 0x2545F4914F6CDD1DULL) >> 40) & 0xFFFFFFu) / 16777216.0f;
}

static int sample_token(IndextContext *ctx, float *logits) {
    int V = (int)ctx->info.vocab_size;

    float rp = ctx->params.repeat_penalty;
    if (rp > 1.0f && ctx->n_recent > 0) {
        int start = ctx->n_recent > 64 ? ctx->n_recent - 64 : 0;
        for (int i = start; i < ctx->n_recent; i++) {
            int tid = ctx->recent_tokens[i];
            if (tid >= 0 && tid < V) {
                if (logits[tid] > 0) logits[tid] /= rp;
                else logits[tid] *= rp;
            }
        }
    }

    float temp = ctx->params.temperature;
    int best = 0;
    float max_l = logits[0];
    for (int i = 1; i < V; i++) {
        if (logits[i] > max_l) { max_l = logits[i]; best = i; }
    }
    if (temp <= 1e-4f) return best;

    float inv_t = 1.0f / temp;
    float thresh = max_l - 15.0f * temp;
    ProbIdx *probs = ctx->probs_buf;
    int n_cand = 0;
    for (int i = 0; i < V; i++) {
        if (logits[i] >= thresh) {
            probs[n_cand].id = i;
            probs[n_cand].prob = logits[i];
            n_cand++;
        }
    }
    if (n_cand <= 1) return best;

    qsort(probs, n_cand, sizeof(ProbIdx), cmp_prob_desc);
    if (ctx->params.top_k > 0 && ctx->params.top_k < n_cand) n_cand = ctx->params.top_k;

    float sum_p = 0.0f;
    for (int i = 0; i < n_cand; i++) {
        float p = fast_expf((probs[i].prob - max_l) * inv_t);
        probs[i].prob = p;
        sum_p += p;
    }
    float inv_sum = 1.0f / sum_p;
    for (int i = 0; i < n_cand; i++) probs[i].prob *= inv_sum;

    float top_p = ctx->params.top_p;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < n_cand; i++) {
            cum += probs[i].prob;
            if (cum >= top_p) { n_cand = i + 1; break; }
        }
    }

    float total = 0.0f;
    for (int i = 0; i < n_cand; i++) total += probs[i].prob;
    float r = rand_f32(&ctx->rng_state) * total;
    float cdf = 0.0f;
    int chosen = probs[0].id;
    for (int i = 0; i < n_cand; i++) {
        cdf += probs[i].prob;
        if (r <= cdf) { chosen = probs[i].id; break; }
    }
    return chosen;
}

int indext_generate(
    IndextContext *ctx,
    const char *prompt,
    int add_bos,
    int continue_ctx,
    IndextTokenCallback cb,
    void *user_data,
    char *err_buf,
    size_t err_len
) {
    if (!ctx || !prompt) return -1;
    ctx->stop_requested = 0;

    if (!continue_ctx) indext_reset_kv(ctx);

    int max_seq = (int)ctx->info.max_seq_len;
    int avail = max_seq - ctx->cur_pos - 1;
    if (avail <= 0) {
        snprintf(err_buf, err_len, "Context is full (%d/%d). Clear context to continue.", ctx->cur_pos, max_seq);
        return -1;
    }
    int *prompt_ids = (int *)malloc(sizeof(int) * avail);
    if (!prompt_ids) {
        snprintf(err_buf, err_len, "Out of memory encoding prompt");
        return -1;
    }
    int use_bos = (add_bos && ctx->info.add_bos_default && ctx->cur_pos == 0) ? 1 : 0;
    int n_prompt = tok_encode(ctx->tok, prompt, use_bos, prompt_ids, avail);
    if (n_prompt <= 0) {
        free(prompt_ids);
        snprintf(err_buf, err_len, "Prompt encoding produced 0 tokens");
        return -1;
    }

    double t_prompt = now_ms();
    int token = prompt_ids[0];
    for (int i = 0; i < n_prompt; i++) {
        if (ctx->stop_requested || ctx->cur_pos >= max_seq - 1) break;
        token = prompt_ids[i];
        int is_last = (i == n_prompt - 1);
        forward_token(ctx, token, ctx->cur_pos, is_last);
        if (ctx->n_recent < max_seq) ctx->recent_tokens[ctx->n_recent++] = token;
        ctx->cur_pos++;
    }
    free(prompt_ids);
    ctx->stats.prompt_tokens = n_prompt;
    ctx->stats.prompt_eval_ms = now_ms() - t_prompt;

    double t_gen = now_ms();
    int gen_count = 0;
    int max_new = ctx->params.max_tokens > 0 ? ctx->params.max_tokens : 128;
    char piece[512];

    while (!ctx->stop_requested && gen_count < max_new && ctx->cur_pos < max_seq - 1) {
        int next = sample_token(ctx, ctx->logits);
        if (next == ctx->info.eos_id || next == tok_eos_id(ctx->tok)) break;

        tok_decode_piece(ctx->tok, next, piece, sizeof(piece));
        gen_count++;
        if (cb && cb(piece, next, user_data) != 0) break;

        if (ctx->n_recent < max_seq) ctx->recent_tokens[ctx->n_recent++] = next;
        forward_token(ctx, next, ctx->cur_pos, 1);
        ctx->cur_pos++;
    }

    ctx->stats.tokens_generated = gen_count;
    ctx->stats.gen_time_ms = now_ms() - t_gen;
    ctx->stats.tok_per_sec = (ctx->stats.gen_time_ms > 0.0) ? ((double)gen_count * 1000.0 / ctx->stats.gen_time_ms) : 0.0;
    return 0;
}
