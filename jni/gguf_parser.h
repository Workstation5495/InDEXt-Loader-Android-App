#ifndef INDEXT_GGUF_PARSER_H
#define INDEXT_GGUF_PARSER_H

#include <stdint.h>
#include <stddef.h>
#include "tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GGML_T_F32  = 0,
    GGML_T_F16  = 1,
    GGML_T_Q4_0 = 2,
    GGML_T_Q4_1 = 3,
    GGML_T_Q5_0 = 6,
    GGML_T_Q5_1 = 7,
    GGML_T_Q8_0 = 8,
    GGML_T_Q2_K = 10,
    GGML_T_Q3_K = 11,
    GGML_T_Q4_K = 12,
    GGML_T_Q5_K = 13,
    GGML_T_Q6_K = 14,
    GGML_T_BF16 = 30
} GgmlType;

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset; /* relative to data_offset */
} GgufTensorInfo;

typedef struct {
    uint32_t version;
    char arch[64];
    char name[128];
    uint32_t alignment;

    uint32_t vocab_size;
    uint32_t dim;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t rope_dim;
    uint32_t hidden_dim;
    uint32_t max_seq_len;
    float rms_eps;
    float rope_freq_base;
    float attn_logit_softcapping;
    float final_logit_softcapping;

    int bos_id;
    int eos_id;
    int unk_id;
    int add_bos; /* -1 = unknown, 0 = false, 1 = true */
    int is_chatml; /* 1 if chat_template uses <|im_start|> */

    uint64_t data_offset;
    uint64_t n_tensors;
    GgufTensorInfo *tensors;

    IndextTokenizer *tokenizer;
} GgufFile;

GgufFile *gguf_parse_memory(const uint8_t *base, size_t size, char *err_buf, size_t err_len);
void gguf_free(GgufFile *gf);
const GgufTensorInfo *gguf_find_tensor(const GgufFile *gf, const char *name);

#ifdef __cplusplus
}
#endif

#endif /* INDEXT_GGUF_PARSER_H */
