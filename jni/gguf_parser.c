#include "gguf_parser.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12
};

typedef struct {
    const uint8_t *p;
    const uint8_t *end;
} BufReader;

static int rd_bytes(BufReader *r, void *out, size_t n) {
    if ((size_t)(r->end - r->p) < n) return -1;
    if (out) memcpy(out, r->p, n);
    r->p += n;
    return 0;
}

static int rd_u32(BufReader *r, uint32_t *v) { return rd_bytes(r, v, 4); }
static int rd_u64(BufReader *r, uint64_t *v) { return rd_bytes(r, v, 8); }
static int rd_f32(BufReader *r, float *v)    { return rd_bytes(r, v, 4); }

static char *rd_str(BufReader *r) {
    uint64_t len = 0;
    if (rd_u64(r, &len) != 0 || len > 16 * 1024 * 1024) return NULL;
    if ((size_t)(r->end - r->p) < (size_t)len) return NULL;
    char *s = (char *)malloc((size_t)len + 1);
    if (!s) return NULL;
    memcpy(s, r->p, (size_t)len);
    s[len] = '\0';
    r->p += (size_t)len;
    return s;
}

static int skip_val(BufReader *r, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:    return rd_bytes(r, NULL, 1);
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:   return rd_bytes(r, NULL, 2);
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return rd_bytes(r, NULL, 4);
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return rd_bytes(r, NULL, 8);
        case GGUF_TYPE_STRING: {
            uint64_t len = 0;
            if (rd_u64(r, &len) != 0) return -1;
            return rd_bytes(r, NULL, (size_t)len);
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t etype = 0;
            uint64_t count = 0;
            if (rd_u32(r, &etype) != 0 || rd_u64(r, &count) != 0) return -1;
            for (uint64_t i = 0; i < count; i++) {
                if (skip_val(r, etype) != 0) return -1;
            }
            return 0;
        }
        default: return -1;
    }
}

static int rd_num_as_u32(BufReader *r, uint32_t type, uint32_t *out) {
    uint8_t u8; int8_t i8; uint16_t u16; int16_t i16;
    uint32_t u32; int32_t i32; uint64_t u64; int64_t i64;
    switch (type) {
        case GGUF_TYPE_UINT8:  if (rd_bytes(r, &u8, 1))  return -1; *out = u8; return 0;
        case GGUF_TYPE_INT8:   if (rd_bytes(r, &i8, 1))  return -1; *out = (uint32_t)i8; return 0;
        case GGUF_TYPE_UINT16: if (rd_bytes(r, &u16, 2)) return -1; *out = u16; return 0;
        case GGUF_TYPE_INT16:  if (rd_bytes(r, &i16, 2)) return -1; *out = (uint32_t)i16; return 0;
        case GGUF_TYPE_UINT32: if (rd_bytes(r, &u32, 4)) return -1; *out = u32; return 0;
        case GGUF_TYPE_INT32:  if (rd_bytes(r, &i32, 4)) return -1; *out = (uint32_t)i32; return 0;
        case GGUF_TYPE_UINT64: if (rd_bytes(r, &u64, 8)) return -1; *out = (uint32_t)u64; return 0;
        case GGUF_TYPE_INT64:  if (rd_bytes(r, &i64, 8)) return -1; *out = (uint32_t)i64; return 0;
        default: return skip_val(r, type);
    }
}

static int rd_num_as_f32(BufReader *r, uint32_t type, float *out) {
    float f32; double f64;
    if (type == GGUF_TYPE_FLOAT32) {
        if (rd_f32(r, &f32)) return -1;
        *out = f32;
        return 0;
    } else if (type == GGUF_TYPE_FLOAT64) {
        if (rd_bytes(r, &f64, 8)) return -1;
        *out = (float)f64;
        return 0;
    }
    return skip_val(r, type);
}

static int ends_with(const char *s, const char *suf) {
    size_t ls = strlen(s), lf = strlen(suf);
    return ls >= lf && strcmp(s + ls - lf, suf) == 0;
}

GgufFile *gguf_parse_memory(const uint8_t *base, size_t size, char *err_buf, size_t err_len) {
    BufReader r = { base, base + size };
    uint8_t magic[4];
    if (rd_bytes(&r, magic, 4) != 0 || memcmp(magic, "GGUF", 4) != 0) {
        snprintf(err_buf, err_len, "Not a valid GGUF file");
        return NULL;
    }

    GgufFile *gf = (GgufFile *)calloc(1, sizeof(GgufFile));
    gf->alignment = 32;
    gf->rms_eps = 1e-5f;
    gf->rope_freq_base = 10000.0f;
    gf->bos_id = 1;
    gf->eos_id = 2;
    gf->unk_id = 0;
    gf->add_bos = -1;
    gf->is_chatml = 0;

    uint64_t n_tensors = 0, n_kv = 0;
    if (rd_u32(&r, &gf->version) != 0 || gf->version < 2 || gf->version > 3) {
        snprintf(err_buf, err_len, "Unsupported GGUF version: %u", gf->version);
        gguf_free(gf);
        return NULL;
    }
    if (rd_u64(&r, &n_tensors) != 0 || rd_u64(&r, &n_kv) != 0) {
        snprintf(err_buf, err_len, "Truncated GGUF header");
        gguf_free(gf);
        return NULL;
    }

    char **tok_list = NULL;
    int tok_count = 0;
    float *tok_scores = NULL;
    char **merge_list = NULL;
    int merge_count = 0;
    IndextTokMode tok_mode = TOK_MODE_SPM;

    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = rd_str(&r);
        uint32_t vtype = 0;
        if (!key || rd_u32(&r, &vtype) != 0) {
            free(key);
            snprintf(err_buf, err_len, "Corrupt GGUF KV header");
            gguf_free(gf);
            return NULL;
        }

        if (strcmp(key, "general.architecture") == 0 && vtype == GGUF_TYPE_STRING) {
            char *s = rd_str(&r);
            if (s) { snprintf(gf->arch, sizeof(gf->arch), "%s", s); free(s); }
        } else if (strcmp(key, "general.name") == 0 && vtype == GGUF_TYPE_STRING) {
            char *s = rd_str(&r);
            if (s) { snprintf(gf->name, sizeof(gf->name), "%s", s); free(s); }
        } else if (strcmp(key, "general.alignment") == 0) {
            rd_num_as_u32(&r, vtype, &gf->alignment);
        } else if (ends_with(key, ".context_length")) {
            rd_num_as_u32(&r, vtype, &gf->max_seq_len);
        } else if (ends_with(key, ".embedding_length")) {
            rd_num_as_u32(&r, vtype, &gf->dim);
        } else if (ends_with(key, ".block_count")) {
            rd_num_as_u32(&r, vtype, &gf->n_layers);
        } else if (ends_with(key, ".feed_forward_length")) {
            rd_num_as_u32(&r, vtype, &gf->hidden_dim);
        } else if (ends_with(key, ".attention.head_count")) {
            rd_num_as_u32(&r, vtype, &gf->n_heads);
        } else if (ends_with(key, ".attention.head_count_kv")) {
            rd_num_as_u32(&r, vtype, &gf->n_kv_heads);
        } else if (ends_with(key, ".attention.key_length")) {
            rd_num_as_u32(&r, vtype, &gf->head_dim);
        } else if (ends_with(key, ".rope.dimension_count")) {
            rd_num_as_u32(&r, vtype, &gf->rope_dim);
        } else if (ends_with(key, ".attention.layer_norm_rms_epsilon")) {
            rd_num_as_f32(&r, vtype, &gf->rms_eps);
        } else if (ends_with(key, ".rope.freq_base")) {
            rd_num_as_f32(&r, vtype, &gf->rope_freq_base);
        } else if (ends_with(key, ".attn_logit_softcapping")) {
            rd_num_as_f32(&r, vtype, &gf->attn_logit_softcapping);
        } else if (ends_with(key, ".final_logit_softcapping")) {
            rd_num_as_f32(&r, vtype, &gf->final_logit_softcapping);
        } else if (strcmp(key, "tokenizer.ggml.model") == 0 && vtype == GGUF_TYPE_STRING) {
            char *s = rd_str(&r);
            if (s) {
                if (strcmp(s, "gpt2") == 0) tok_mode = TOK_MODE_GPT2_BPE;
                else tok_mode = TOK_MODE_SPM;
                free(s);
            }
        } else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
            uint32_t v = 1; rd_num_as_u32(&r, vtype, &v); gf->bos_id = (int)v;
        } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
            uint32_t v = 2; rd_num_as_u32(&r, vtype, &v); gf->eos_id = (int)v;
        } else if (strcmp(key, "tokenizer.ggml.unknown_token_id") == 0) {
            uint32_t v = 0; rd_num_as_u32(&r, vtype, &v); gf->unk_id = (int)v;
        } else if (strcmp(key, "tokenizer.ggml.add_bos_token") == 0 && vtype == GGUF_TYPE_BOOL) {
            uint8_t b = 1; rd_bytes(&r, &b, 1); gf->add_bos = b ? 1 : 0;
        } else if (strcmp(key, "tokenizer.chat_template") == 0 && vtype == GGUF_TYPE_STRING) {
            char *s = rd_str(&r);
            if (s) {
                if (strstr(s, "<|im_start|>") != NULL) gf->is_chatml = 1;
                free(s);
            }
        } else if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vtype == GGUF_TYPE_ARRAY) {
            uint32_t et = 0; uint64_t cnt = 0;
            rd_u32(&r, &et); rd_u64(&r, &cnt);
            if (et == GGUF_TYPE_STRING && cnt > 0 && cnt < 1000000) {
                tok_count = (int)cnt;
                tok_list = (char **)calloc(tok_count, sizeof(char *));
                for (int k = 0; k < tok_count; k++) tok_list[k] = rd_str(&r);
            } else {
                for (uint64_t k = 0; k < cnt; k++) skip_val(&r, et);
            }
        } else if (strcmp(key, "tokenizer.ggml.scores") == 0 && vtype == GGUF_TYPE_ARRAY) {
            uint32_t et = 0; uint64_t cnt = 0;
            rd_u32(&r, &et); rd_u64(&r, &cnt);
            if (et == GGUF_TYPE_FLOAT32 && cnt > 0 && cnt < 1000000) {
                tok_scores = (float *)calloc((size_t)cnt, sizeof(float));
                for (uint64_t k = 0; k < cnt; k++) rd_f32(&r, &tok_scores[k]);
            } else {
                for (uint64_t k = 0; k < cnt; k++) skip_val(&r, et);
            }
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0 && vtype == GGUF_TYPE_ARRAY) {
            uint32_t et = 0; uint64_t cnt = 0;
            rd_u32(&r, &et); rd_u64(&r, &cnt);
            if (et == GGUF_TYPE_STRING && cnt > 0 && cnt < 2000000) {
                merge_count = (int)cnt;
                merge_list = (char **)calloc(merge_count, sizeof(char *));
                for (int k = 0; k < merge_count; k++) merge_list[k] = rd_str(&r);
            } else {
                for (uint64_t k = 0; k < cnt; k++) skip_val(&r, et);
            }
        } else {
            if (skip_val(&r, vtype) != 0) {
                free(key);
                snprintf(err_buf, err_len, "Failed skipping GGUF KV");
                gguf_free(gf);
                return NULL;
            }
        }
        free(key);
    }

    if (gf->n_kv_heads == 0) gf->n_kv_heads = gf->n_heads;
    if (gf->vocab_size == 0 && tok_count > 0) gf->vocab_size = (uint32_t)tok_count;
    if (gf->alignment == 0) gf->alignment = 32;

    gf->n_tensors = n_tensors;
    gf->tensors = (GgufTensorInfo *)calloc((size_t)n_tensors, sizeof(GgufTensorInfo));
    for (uint64_t i = 0; i < n_tensors; i++) {
        GgufTensorInfo *ti = &gf->tensors[i];
        ti->name = rd_str(&r);
        if (!ti->name || rd_u32(&r, &ti->n_dims) != 0 || ti->n_dims > 4) {
            snprintf(err_buf, err_len, "Invalid GGUF tensor info");
            gguf_free(gf);
            return NULL;
        }
        for (uint32_t d = 0; d < ti->n_dims; d++) rd_u64(&r, &ti->dims[d]);
        rd_u32(&r, &ti->type);
        rd_u64(&r, &ti->offset);
    }

    uint64_t pos = (uint64_t)(r.p - base);
    uint64_t align = gf->alignment;
    gf->data_offset = (pos + align - 1) & ~(align - 1);

    if (tok_list && tok_count > 0) {
        gf->tokenizer = tok_create_from_arrays(
            tok_mode, tok_count,
            (const char **)tok_list, tok_scores,
            merge_count, (const char **)merge_list,
            gf->bos_id, gf->eos_id, gf->unk_id,
            err_buf, err_len
        );
    }
    if (tok_list) { for (int i = 0; i < tok_count; i++) free(tok_list[i]); free(tok_list); }
    free(tok_scores);
    if (merge_list) { for (int i = 0; i < merge_count; i++) free(merge_list[i]); free(merge_list); }

    return gf;
}

void gguf_free(GgufFile *gf) {
    if (!gf) return;
    if (gf->tensors) {
        for (uint64_t i = 0; i < gf->n_tensors; i++) free(gf->tensors[i].name);
        free(gf->tensors);
    }
    tok_free(gf->tokenizer);
    free(gf);
}

const GgufTensorInfo *gguf_find_tensor(const GgufFile *gf, const char *name) {
    if (!gf || !name) return NULL;
    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        if (gf->tensors[i].name && strcmp(gf->tensors[i].name, name) == 0) {
            return &gf->tensors[i];
        }
    }
    return NULL;
}
