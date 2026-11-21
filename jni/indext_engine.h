#ifndef INDEXT_ENGINE_H
#define INDEXT_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    INDEXT_FMT_UNKNOWN = 0,
    INDEXT_FMT_IDXT    = 1,
    INDEXT_FMT_GGUF    = 2
} IndextFormat;

typedef struct {
    int n_threads;        /* 0 = auto */
    int n_ctx;            /* 0 = model default */
    int max_tokens;       /* max new tokens per turn */
    float temperature;    /* 0.0 = greedy */
    float top_p;          /* 0.0..1.0 */
    int top_k;            /* <=0 disabled */
    float repeat_penalty; /* 1.0 = disabled */
    int seed;             /* 0 = time-based */
    int use_mmap;         /* 1 = mmap (zero-copy), 0 = read into RAM */
    int kv_mode;          /* 0 = FP32, 1 = FP16, 2 = Q8_0, 3 = Q4_0 */
} IndextParams;

typedef struct {
    IndextFormat format;
    char arch[64];
    char model_name[128];
    uint32_t vocab_size;
    uint32_t dim;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t hidden_dim;
    uint32_t head_dim;
    uint32_t max_seq_len;
    float rms_eps;
    float rope_freq_base;
    int bos_id;
    int eos_id;
    int add_bos_default;
    int is_chatml;
    int idxt_version;
    int used_mmap;
    int kv_mode;
    uint32_t quant_type;
    size_t file_size;
    size_t kv_cache_bytes;
} IndextModelInfo;

typedef struct {
    int tokens_generated;
    int prompt_tokens;
    double load_time_ms;
    double prompt_eval_ms;
    double gen_time_ms;
    double tok_per_sec;
} IndextStats;

/* Return 0 to continue, non-zero to stop generation */
typedef int (*IndextTokenCallback)(const char *piece, int token_id, void *user_data);

typedef struct IndextContext IndextContext;

IndextParams indext_default_params(void);

/* Load IDXT (v1/v2, requires tokenizer_path) or GGUF (tokenizer_path may be NULL). */
IndextContext *indext_load(
    const char *model_path,
    const char *tokenizer_path,
    const IndextParams *params,
    char *err_buf,
    size_t err_buf_len
);

void indext_free(IndextContext *ctx);
void indext_reset_kv(IndextContext *ctx);
void indext_get_info(const IndextContext *ctx, IndextModelInfo *out_info);
void indext_get_stats(const IndextContext *ctx, IndextStats *out_stats);
void indext_update_params(IndextContext *ctx, const IndextParams *params);
void indext_request_stop(IndextContext *ctx);

/* Generate from raw prompt (appends to existing KV cache if continue_ctx != 0). */
int indext_generate(
    IndextContext *ctx,
    const char *prompt,
    int add_bos,
    int continue_ctx,
    IndextTokenCallback cb,
    void *user_data,
    char *err_buf,
    size_t err_buf_len
);

#ifdef __cplusplus
}
#endif

#endif /* INDEXT_ENGINE_H */
