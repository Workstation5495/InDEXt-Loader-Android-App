#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct {
    char *str;
    int len;
    float score;
    int is_special;
} TokEntry;

typedef struct {
    char *key;
    int key_len;
    int val;
    uint32_t hash;
} StrIntSlot;

struct IndextTokenizer {
    IndextTokMode mode;
    int vocab_size;
    TokEntry *entries;

    /* Hash map: token string -> token id */
    StrIntSlot *vocab_map;
    size_t vocab_cap;

    /* Hash map: "tokA tokB" -> merge rank (lower = higher priority) */
    StrIntSlot *merge_map;
    size_t merge_cap;
    int n_merges;

    /* Special tokens sorted by length descending for pre-matching */
    int *special_ids;
    int n_special;

    int bos_id;
    int eos_id;
    int unk_id;
};

static uint32_t fnv1a(const char *s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h ? h : 1u;
}

static size_t next_pow2(size_t n) {
    size_t p = 64;
    while (p < n * 2) p <<= 1;
    return p;
}

static void map_insert(StrIntSlot *slots, size_t cap, const char *key, int key_len, int val, int own_key) {
    uint32_t h = fnv1a(key, key_len);
    size_t mask = cap - 1;
    size_t idx = (size_t)h & mask;
    while (slots[idx].key != NULL) {
        if (slots[idx].hash == h && slots[idx].key_len == key_len && memcmp(slots[idx].key, key, key_len) == 0) {
            slots[idx].val = val;
            return;
        }
        idx = (idx + 1) & mask;
    }
    if (own_key) {
        char *copy = (char *)malloc(key_len + 1);
        memcpy(copy, key, key_len);
        copy[key_len] = '\0';
        slots[idx].key = copy;
    } else {
        slots[idx].key = (char *)key;
    }
    slots[idx].key_len = key_len;
    slots[idx].val = val;
    slots[idx].hash = h;
}

static int map_find(const StrIntSlot *slots, size_t cap, const char *key, int key_len) {
    if (!slots || cap == 0) return -1;
    uint32_t h = fnv1a(key, key_len);
    size_t mask = cap - 1;
    size_t idx = (size_t)h & mask;
    while (slots[idx].key != NULL) {
        if (slots[idx].hash == h && slots[idx].key_len == key_len && memcmp(slots[idx].key, key, key_len) == 0) {
            return slots[idx].val;
        }
        idx = (idx + 1) & mask;
    }
    return -1;
}

/* GPT-2 byte-to-unicode mapping (bytes 0..255 -> UTF-8 string) */
static void gpt2_byte_to_utf8(uint8_t b, char out[4], int *out_len) {
    uint32_t cp;
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
        cp = b;
    } else {
        /* 256 + index in list of non-printable/space bytes */
        int idx = 0;
        for (int i = 0; i < (int)b; i++) {
            if (!((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255))) {
                idx++;
            }
        }
        cp = 256 + idx;
    }
    if (cp < 0x80) {
        out[0] = (char)cp;
        *out_len = 1;
    } else {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        *out_len = 2;
    }
}

static int gpt2_utf8_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
        return (int)cp;
    }
    if (cp >= 256 && cp < 512) {
        int target = (int)(cp - 256);
        int idx = 0;
        for (int i = 0; i < 256; i++) {
            if (!((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255))) {
                if (idx == target) return i;
                idx++;
            }
        }
    }
    return -1;
}

static int utf8_decode_cp(const char *s, int len, uint32_t *cp) {
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80) {
        *cp = c;
        return 1;
    } else if ((c & 0xE0) == 0xC0 && len >= 2) {
        *cp = ((uint32_t)(c & 0x1F) << 6) | (uint32_t)(s[1] & 0x3F);
        return 2;
    } else if ((c & 0xF0) == 0xE0 && len >= 3) {
        *cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (uint32_t)(s[2] & 0x3F);
        return 3;
    } else if ((c & 0xF8) == 0xF0 && len >= 4) {
        *cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) | ((uint32_t)(s[2] & 0x3F) << 6) | (uint32_t)(s[3] & 0x3F);
        return 4;
    }
    *cp = c;
    return 1;
}

static int utf8_encode_cp(uint32_t cp, char *out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
}

/* Minimal JSON string parser helpers for HuggingFace tokenizer.json */
static const char *skip_ws(const char *p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) p++;
    return p;
}

static int parse_hex4(const char *p, uint32_t *out) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else return -1;
    }
    *out = v;
    return 0;
}

/* Parses a JSON string starting at *pp (pointing to '"'). Returns malloc'd UTF-8 string. */
static char *parse_json_str(const char **pp, int *out_len) {
    const char *p = *pp;
    if (*p != '"') return NULL;
    p++;
    size_t cap = 64, len = 0;
    char *buf = (char *)malloc(cap);
    while (*p && *p != '"') {
        if (len + 8 >= cap) {
            cap *= 2;
            buf = (char *)realloc(buf, cap);
        }
        if (*p == '\\') {
            p++;
            char c = *p++;
            if (c == '"' || c == '\\' || c == '/') buf[len++] = c;
            else if (c == 'b') buf[len++] = '\b';
            else if (c == 'f') buf[len++] = '\f';
            else if (c == 'n') buf[len++] = '\n';
            else if (c == 'r') buf[len++] = '\r';
            else if (c == 't') buf[len++] = '\t';
            else if (c == 'u') {
                uint32_t cp = 0;
                if (parse_hex4(p, &cp) == 0) {
                    p += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && p[0] == '\\' && p[1] == 'u') {
                        uint32_t low = 0;
                        if (parse_hex4(p + 2, &low) == 0 && low >= 0xDC00 && low <= 0xDFFF) {
                            p += 6;
                            cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                        }
                    }
                    len += utf8_encode_cp(cp, buf + len);
                }
            } else {
                buf[len++] = c;
            }
        } else {
            buf[len++] = *p++;
        }
    }
    if (*p == '"') p++;
    buf[len] = '\0';
    *pp = p;
    if (out_len) *out_len = (int)len;
    return buf;
}

static const char *skip_json_value(const char *p) {
    p = skip_ws(p);
    if (*p == '"') {
        p++;
        while (*p && *p != '"') {
            if (*p == '\\' && p[1]) p += 2;
            else p++;
        }
        if (*p == '"') p++;
        return p;
    } else if (*p == '{' || *p == '[') {
        char open = *p, close = (open == '{') ? '}' : ']';
        int depth = 1;
        p++;
        while (*p && depth > 0) {
            if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\' && p[1]) p += 2;
                    else p++;
                }
                if (*p == '"') p++;
            } else {
                if (*p == open) depth++;
                else if (*p == close) depth--;
                p++;
            }
        }
        return p;
    } else {
        while (*p && *p != ',' && *p != '}' && *p != ']' && !isspace((unsigned char)*p)) p++;
        return p;
    }
}

static void build_maps_and_specials(IndextTokenizer *tok) {
    tok->vocab_cap = next_pow2((size_t)tok->vocab_size + 16);
    tok->vocab_map = (StrIntSlot *)calloc(tok->vocab_cap, sizeof(StrIntSlot));
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->entries[i].str) {
            map_insert(tok->vocab_map, tok->vocab_cap, tok->entries[i].str, tok->entries[i].len, i, 0);
        }
    }
    /* Collect special tokens sorted by length desc */
    int cnt = 0;
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->entries[i].is_special) cnt++;
    }
    tok->n_special = cnt;
    if (cnt > 0) {
        tok->special_ids = (int *)malloc(sizeof(int) * cnt);
        int idx = 0;
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->entries[i].is_special) tok->special_ids[idx++] = i;
        }
        for (int i = 0; i < cnt - 1; i++) {
            for (int j = i + 1; j < cnt; j++) {
                if (tok->entries[tok->special_ids[j]].len > tok->entries[tok->special_ids[i]].len) {
                    int tmp = tok->special_ids[i];
                    tok->special_ids[i] = tok->special_ids[j];
                    tok->special_ids[j] = tmp;
                }
            }
        }
    }
}

IndextTokenizer *tok_create_from_arrays(
    IndextTokMode mode,
    int vocab_size,
    const char **tokens,
    const float *scores,
    int n_merges,
    const char **merges,
    int bos_id,
    int eos_id,
    int unk_id,
    char *err_buf,
    size_t err_len
) {
    if (vocab_size <= 0 || !tokens) {
        snprintf(err_buf, err_len, "Invalid vocabulary arrays");
        return NULL;
    }
    IndextTokenizer *tok = (IndextTokenizer *)calloc(1, sizeof(IndextTokenizer));
    tok->mode = mode;
    tok->vocab_size = vocab_size;
    tok->entries = (TokEntry *)calloc(vocab_size, sizeof(TokEntry));
    tok->bos_id = bos_id;
    tok->eos_id = eos_id;
    tok->unk_id = unk_id;

    for (int i = 0; i < vocab_size; i++) {
        const char *s = tokens[i] ? tokens[i] : "";
        int l = (int)strlen(s);
        tok->entries[i].str = (char *)malloc(l + 1);
        memcpy(tok->entries[i].str, s, l + 1);
        tok->entries[i].len = l;
        tok->entries[i].score = scores ? scores[i] : (float)(-i);
        if (i == bos_id || i == eos_id || (l >= 4 && s[0] == '<' && s[l - 1] == '>')) {
            /* Mark angle-bracket special tokens like <|user|>, <bos>, <eos>, <s>, </s> (except byte <0xXX>) */
            if (!(l == 6 && strncmp(s, "<0x", 3) == 0)) {
                tok->entries[i].is_special = 1;
            }
        }
    }

    tok->n_merges = n_merges;
    if (n_merges > 0 && merges) {
        tok->merge_cap = next_pow2((size_t)n_merges + 16);
        tok->merge_map = (StrIntSlot *)calloc(tok->merge_cap, sizeof(StrIntSlot));
        for (int i = 0; i < n_merges; i++) {
            if (merges[i]) {
                map_insert(tok->merge_map, tok->merge_cap, merges[i], (int)strlen(merges[i]), i, 1);
            }
        }
    }

    build_maps_and_specials(tok);
    return tok;
}

IndextTokenizer *tok_load_json(const char *path, char *err_buf, size_t err_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(err_buf, err_len, "Cannot open tokenizer: %s", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsz <= 0 || fsz > 64 * 1024 * 1024) {
        fclose(f);
        snprintf(err_buf, err_len, "Invalid tokenizer file size");
        return NULL;
    }
    char *json = (char *)malloc((size_t)fsz + 1);
    if (!json) {
        fclose(f);
        snprintf(err_buf, err_len, "Out of memory reading tokenizer");
        return NULL;
    }
    size_t nr = fread(json, 1, (size_t)fsz, f);
    fclose(f);
    json[nr] = '\0';

    IndextTokenizer *tok = (IndextTokenizer *)calloc(1, sizeof(IndextTokenizer));
    tok->mode = TOK_MODE_GPT2_BPE;
    tok->bos_id = -1;
    tok->eos_id = -1;
    tok->unk_id = 0;

    /* First pass: parse added_tokens and model.{vocab, merges} */
    int max_id = -1;
    int cap_entries = 16384;
    tok->entries = (TokEntry *)calloc(cap_entries, sizeof(TokEntry));

    int merges_cap = 16384;
    char **merges_tmp = (char **)malloc(sizeof(char *) * merges_cap);
    int n_merges = 0;

    const char *p = skip_ws(json);
    if (*p != '{') {
        free(json);
        free(merges_tmp);
        tok_free(tok);
        snprintf(err_buf, err_len, "Tokenizer JSON must be an object");
        return NULL;
    }
    p++;
    while (*p) {
        p = skip_ws(p);
        if (*p == '}') break;
        if (*p == ',') { p++; continue; }
        int klen = 0;
        char *key = parse_json_str(&p, &klen);
        if (!key) break;
        p = skip_ws(p);
        if (*p == ':') p++;
        p = skip_ws(p);

        if (strcmp(key, "added_tokens") == 0 && *p == '[') {
            p++;
            while (*p) {
                p = skip_ws(p);
                if (*p == ']') { p++; break; }
                if (*p == ',') { p++; continue; }
                if (*p == '{') {
                    p++;
                    int id = -1;
                    char *content = NULL;
                    int content_len = 0;
                    int is_spec = 1;
                    while (*p) {
                        p = skip_ws(p);
                        if (*p == '}') { p++; break; }
                        if (*p == ',') { p++; continue; }
                        char *ak = parse_json_str(&p, NULL);
                        if (!ak) break;
                        p = skip_ws(p);
                        if (*p == ':') p++;
                        p = skip_ws(p);
                        if (strcmp(ak, "id") == 0) {
                            id = (int)strtol(p, (char **)&p, 10);
                        } else if (strcmp(ak, "content") == 0) {
                            content = parse_json_str(&p, &content_len);
                        } else if (strcmp(ak, "special") == 0) {
                            is_spec = (strncmp(p, "true", 4) == 0);
                            p = skip_json_value(p);
                        } else {
                            p = skip_json_value(p);
                        }
                        free(ak);
                    }
                    if (id >= 0 && content) {
                        while (id >= cap_entries) {
                            int old = cap_entries;
                            cap_entries *= 2;
                            tok->entries = (TokEntry *)realloc(tok->entries, sizeof(TokEntry) * cap_entries);
                            memset(tok->entries + old, 0, sizeof(TokEntry) * (cap_entries - old));
                        }
                        free(tok->entries[id].str);
                        tok->entries[id].str = content;
                        tok->entries[id].len = content_len;
                        tok->entries[id].is_special = is_spec;
                        if (id > max_id) max_id = id;
                        if (strcmp(content, "<bos>") == 0 || strcmp(content, "<s>") == 0) tok->bos_id = id;
                        if (strcmp(content, "<eos>") == 0 || strcmp(content, "</s>") == 0) tok->eos_id = id;
                        if (strcmp(content, "<unk>") == 0) tok->unk_id = id;
                    } else {
                        free(content);
                    }
                } else {
                    p = skip_json_value(p);
                }
            }
        } else if (strcmp(key, "model") == 0 && *p == '{') {
            p++;
            while (*p) {
                p = skip_ws(p);
                if (*p == '}') { p++; break; }
                if (*p == ',') { p++; continue; }
                char *mk = parse_json_str(&p, NULL);
                if (!mk) break;
                p = skip_ws(p);
                if (*p == ':') p++;
                p = skip_ws(p);

                if (strcmp(mk, "vocab") == 0 && *p == '{') {
                    p++;
                    while (*p) {
                        p = skip_ws(p);
                        if (*p == '}') { p++; break; }
                        if (*p == ',') { p++; continue; }
                        int tlen = 0;
                        char *tstr = parse_json_str(&p, &tlen);
                        if (!tstr) break;
                        p = skip_ws(p);
                        if (*p == ':') p++;
                        p = skip_ws(p);
                        int tid = (int)strtol(p, (char **)&p, 10);
                        if (tid >= 0) {
                            while (tid >= cap_entries) {
                                int old = cap_entries;
                                cap_entries *= 2;
                                tok->entries = (TokEntry *)realloc(tok->entries, sizeof(TokEntry) * cap_entries);
                                memset(tok->entries + old, 0, sizeof(TokEntry) * (cap_entries - old));
                            }
                            if (!tok->entries[tid].str) {
                                tok->entries[tid].str = tstr;
                                tok->entries[tid].len = tlen;
                            } else {
                                free(tstr);
                            }
                            if (tid > max_id) max_id = tid;
                        } else {
                            free(tstr);
                        }
                    }
                } else if (strcmp(mk, "merges") == 0 && *p == '[') {
                    p++;
                    while (*p) {
                        p = skip_ws(p);
                        if (*p == ']') { p++; break; }
                        if (*p == ',') { p++; continue; }
                        char *mstr = NULL;
                        if (*p == '"') {
                            mstr = parse_json_str(&p, NULL);
                        } else if (*p == '[') {
                            /* Some tokenizers store merges as ["a", "b"] */
                            p++;
                            p = skip_ws(p);
                            int l1 = 0, l2 = 0;
                            char *s1 = parse_json_str(&p, &l1);
                            p = skip_ws(p);
                            if (*p == ',') p++;
                            p = skip_ws(p);
                            char *s2 = parse_json_str(&p, &l2);
                            p = skip_ws(p);
                            if (*p == ']') p++;
                            if (s1 && s2) {
                                mstr = (char *)malloc(l1 + l2 + 2);
                                memcpy(mstr, s1, l1);
                                mstr[l1] = ' ';
                                memcpy(mstr + l1 + 1, s2, l2 + 1);
                            }
                            free(s1);
                            free(s2);
                        } else {
                            p = skip_json_value(p);
                        }
                        if (mstr) {
                            if (n_merges >= merges_cap) {
                                merges_cap *= 2;
                                merges_tmp = (char **)realloc(merges_tmp, sizeof(char *) * merges_cap);
                            }
                            merges_tmp[n_merges++] = mstr;
                        }
                    }
                } else {
                    p = skip_json_value(p);
                }
                free(mk);
            }
        } else {
            p = skip_json_value(p);
        }
        free(key);
    }

    free(json);
    tok->vocab_size = max_id + 1;
    if (tok->vocab_size <= 0) {
        for (int i = 0; i < n_merges; i++) free(merges_tmp[i]);
        free(merges_tmp);
        tok_free(tok);
        snprintf(err_buf, err_len, "Empty vocabulary in tokenizer.json");
        return NULL;
    }

    tok->n_merges = n_merges;
    if (n_merges > 0) {
        tok->merge_cap = next_pow2((size_t)n_merges + 16);
        tok->merge_map = (StrIntSlot *)calloc(tok->merge_cap, sizeof(StrIntSlot));
        for (int i = 0; i < n_merges; i++) {
            map_insert(tok->merge_map, tok->merge_cap, merges_tmp[i], (int)strlen(merges_tmp[i]), i, 0);
        }
    }
    free(merges_tmp);

    build_maps_and_specials(tok);
    return tok;
}

void tok_free(IndextTokenizer *tok) {
    if (!tok) return;
    if (tok->entries) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->entries[i].str);
        free(tok->entries);
    }
    free(tok->vocab_map);
    if (tok->merge_map) {
        for (size_t i = 0; i < tok->merge_cap; i++) free(tok->merge_map[i].key);
        free(tok->merge_map);
    }
    free(tok->special_ids);
    free(tok);
}

int tok_vocab_size(const IndextTokenizer *tok) { return tok ? tok->vocab_size : 0; }
int tok_bos_id(const IndextTokenizer *tok) { return tok ? tok->bos_id : -1; }
int tok_eos_id(const IndextTokenizer *tok) { return tok ? tok->eos_id : -1; }
int tok_unk_id(const IndextTokenizer *tok) { return tok ? tok->unk_id : 0; }

/* Encode a non-special text segment using BPE */
static int encode_bpe_segment(const IndextTokenizer *tok, const char *seg, int seg_len, int *out_ids, int max_ids) {
    if (seg_len <= 0) return 0;

    typedef struct {
        char *str;
        int len;
    } Sym;

    int max_syms = seg_len * 2 + 8;
    Sym *syms = (Sym *)malloc(sizeof(Sym) * max_syms);
    int n_syms = 0;

    if (tok->mode == TOK_MODE_GPT2_BPE) {
        for (int i = 0; i < seg_len; i++) {
            char buf[4];
            int blen = 0;
            gpt2_byte_to_utf8((uint8_t)seg[i], buf, &blen);
            syms[n_syms].str = (char *)malloc(blen + 1);
            memcpy(syms[n_syms].str, buf, blen);
            syms[n_syms].str[blen] = '\0';
            syms[n_syms].len = blen;
            n_syms++;
        }
    } else {
        /* SentencePiece: replace spaces with U+2581 (\xE2\x96\x81) and split into UTF-8 chars */
        int i = 0;
        while (i < seg_len) {
            if (seg[i] == ' ') {
                syms[n_syms].str = (char *)malloc(4);
                memcpy(syms[n_syms].str, "\xE2\x96\x81", 4);
                syms[n_syms].len = 3;
                n_syms++;
                i++;
            } else {
                uint32_t cp = 0;
                int clen = utf8_decode_cp(seg + i, seg_len - i, &cp);
                syms[n_syms].str = (char *)malloc(clen + 1);
                memcpy(syms[n_syms].str, seg + i, clen);
                syms[n_syms].str[clen] = '\0';
                syms[n_syms].len = clen;
                n_syms++;
                i += clen;
            }
        }
    }

    char *pair_buf = (char *)malloc(4096);
    while (n_syms >= 2) {
        int best_idx = -1;
        int best_rank = 0x7fffffff;
        float best_score = -1e30f;

        for (int i = 0; i < n_syms - 1; i++) {
            int l1 = syms[i].len, l2 = syms[i + 1].len;
            if (l1 + l2 + 2 >= 4096) continue;
            if (tok->merge_map) {
                memcpy(pair_buf, syms[i].str, l1);
                pair_buf[l1] = ' ';
                memcpy(pair_buf + l1 + 1, syms[i + 1].str, l2);
                pair_buf[l1 + 1 + l2] = '\0';
                int rank = map_find(tok->merge_map, tok->merge_cap, pair_buf, l1 + 1 + l2);
                if (rank >= 0 && rank < best_rank) {
                    best_rank = rank;
                    best_idx = i;
                }
            } else {
                memcpy(pair_buf, syms[i].str, l1);
                memcpy(pair_buf + l1, syms[i + 1].str, l2);
                pair_buf[l1 + l2] = '\0';
                int id = map_find(tok->vocab_map, tok->vocab_cap, pair_buf, l1 + l2);
                if (id >= 0 && tok->entries[id].score > best_score) {
                    best_score = tok->entries[id].score;
                    best_idx = i;
                }
            }
        }

        if (best_idx < 0) break;

        int nl = syms[best_idx].len + syms[best_idx + 1].len;
        char *merged = (char *)malloc(nl + 1);
        memcpy(merged, syms[best_idx].str, syms[best_idx].len);
        memcpy(merged + syms[best_idx].len, syms[best_idx + 1].str, syms[best_idx + 1].len);
        merged[nl] = '\0';

        free(syms[best_idx].str);
        free(syms[best_idx + 1].str);
        syms[best_idx].str = merged;
        syms[best_idx].len = nl;

        for (int j = best_idx + 1; j < n_syms - 1; j++) {
            syms[j] = syms[j + 1];
        }
        n_syms--;
    }
    free(pair_buf);

    int out_n = 0;
    for (int i = 0; i < n_syms; i++) {
        int id = map_find(tok->vocab_map, tok->vocab_cap, syms[i].str, syms[i].len);
        if (id >= 0) {
            if (out_n < max_ids) out_ids[out_n++] = id;
        } else if (tok->mode == TOK_MODE_SPM) {
            for (int b = 0; b < syms[i].len; b++) {
                char hex[8];
                snprintf(hex, sizeof(hex), "<0x%02X>", (uint8_t)syms[i].str[b]);
                int bid = map_find(tok->vocab_map, tok->vocab_cap, hex, 6);
                if (out_n < max_ids) out_ids[out_n++] = (bid >= 0) ? bid : tok->unk_id;
            }
        } else {
            if (out_n < max_ids) out_ids[out_n++] = tok->unk_id;
        }
        free(syms[i].str);
    }
    free(syms);
    return out_n;
}

int tok_encode(const IndextTokenizer *tok, const char *text, int add_bos, int *out_ids, int max_ids) {
    if (!tok || !text || !out_ids || max_ids <= 0) return -1;
    int count = 0;
    if (add_bos && tok->bos_id >= 0 && count < max_ids) {
        out_ids[count++] = tok->bos_id;
    }

    int len = (int)strlen(text);
    int pos = 0;
    int seg_start = 0;

    while (pos < len) {
        int matched_id = -1;
        int matched_len = 0;
        for (int s = 0; s < tok->n_special; s++) {
            int sid = tok->special_ids[s];
            int slen = tok->entries[sid].len;
            if (slen > 0 && pos + slen <= len && memcmp(text + pos, tok->entries[sid].str, slen) == 0) {
                matched_id = sid;
                matched_len = slen;
                break;
            }
        }
        if (matched_id >= 0) {
            if (pos > seg_start) {
                if (tok->mode == TOK_MODE_GPT2_BPE) {
                    /* Split into GPT-2 word chunks (leading space attaches to next word, newlines split) */
                    int p = seg_start;
                    while (p < pos) {
                        int cs = p;
                        if (text[p] == '\n' || text[p] == '\r') {
                            p++;
                        } else {
                            if (text[p] == ' ') p++;
                            while (p < pos && text[p] != ' ' && text[p] != '\n' && text[p] != '\r') p++;
                        }
                        count += encode_bpe_segment(tok, text + cs, p - cs, out_ids + count, max_ids - count);
                    }
                } else {
                    count += encode_bpe_segment(tok, text + seg_start, pos - seg_start, out_ids + count, max_ids - count);
                }
            }
            if (count < max_ids) out_ids[count++] = matched_id;
            pos += matched_len;
            seg_start = pos;
        } else {
            pos++;
        }
    }
    if (pos > seg_start) {
        if (tok->mode == TOK_MODE_GPT2_BPE) {
            int p = seg_start;
            while (p < pos) {
                int cs = p;
                if (text[p] == '\n' || text[p] == '\r') {
                    p++;
                } else {
                    if (text[p] == ' ') p++;
                    while (p < pos && text[p] != ' ' && text[p] != '\n' && text[p] != '\r') p++;
                }
                count += encode_bpe_segment(tok, text + cs, p - cs, out_ids + count, max_ids - count);
            }
        } else {
            count += encode_bpe_segment(tok, text + seg_start, pos - seg_start, out_ids + count, max_ids - count);
        }
    }
    return count;
}

int tok_decode_piece(const IndextTokenizer *tok, int token_id, char *out_buf, size_t out_cap) {
    if (!tok || !out_buf || out_cap == 0) return 0;
    out_buf[0] = '\0';
    if (token_id < 0 || token_id >= tok->vocab_size || !tok->entries[token_id].str) return 0;

    const TokEntry *e = &tok->entries[token_id];
    if (e->is_special) {
        if (token_id == tok->bos_id || token_id == tok->eos_id) return 0;
        size_t n = (size_t)e->len < out_cap - 1 ? (size_t)e->len : out_cap - 1;
        memcpy(out_buf, e->str, n);
        out_buf[n] = '\0';
        return (int)n;
    }

    if (tok->mode == TOK_MODE_SPM) {
        if (e->len == 6 && strncmp(e->str, "<0x", 3) == 0 && e->str[5] == '>') {
            unsigned int byte_val = 0;
            if (sscanf(e->str + 3, "%02X", &byte_val) == 1 && out_cap >= 2) {
                out_buf[0] = (char)byte_val;
                out_buf[1] = '\0';
                return 1;
            }
        }
        size_t out_i = 0;
        int i = 0;
        while (i < e->len && out_i + 1 < out_cap) {
            if (i + 3 <= e->len && memcmp(e->str + i, "\xE2\x96\x81", 3) == 0) {
                out_buf[out_i++] = ' ';
                i += 3;
            } else {
                out_buf[out_i++] = e->str[i++];
            }
        }
        out_buf[out_i] = '\0';
        return (int)out_i;
    }

    /* GPT-2 ByteLevel decode */
    size_t out_i = 0;
    int i = 0;
    while (i < e->len && out_i + 1 < out_cap) {
        uint32_t cp = 0;
        int adv = utf8_decode_cp(e->str + i, e->len - i, &cp);
        int b = gpt2_utf8_to_byte(cp);
        if (b >= 0) {
            out_buf[out_i++] = (char)b;
        } else {
            for (int k = 0; k < adv && out_i + 1 < out_cap; k++) {
                out_buf[out_i++] = e->str[i + k];
            }
        }
        i += adv;
    }
    out_buf[out_i] = '\0';
    return (int)out_i;
}
