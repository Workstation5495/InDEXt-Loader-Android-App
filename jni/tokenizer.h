#ifndef INDEXT_TOKENIZER_H
#define INDEXT_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TOK_MODE_GPT2_BPE = 0,  /* ByteLevel BPE (HuggingFace tokenizer.json / GGUF gpt2) */
    TOK_MODE_SPM      = 1   /* SentencePiece-style ('\xE2\x96\x81' space + <0xXX> byte fallback) */
} IndextTokMode;

typedef struct IndextTokenizer IndextTokenizer;

IndextTokenizer *tok_load_json(const char *path, char *err_buf, size_t err_len);

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
);

void tok_free(IndextTokenizer *tok);

int tok_vocab_size(const IndextTokenizer *tok);
int tok_bos_id(const IndextTokenizer *tok);
int tok_eos_id(const IndextTokenizer *tok);
int tok_unk_id(const IndextTokenizer *tok);

/* Encode UTF-8 text into token IDs. Returns number of tokens written, or -1 on error. */
int tok_encode(
    const IndextTokenizer *tok,
    const char *text,
    int add_bos,
    int *out_ids,
    int max_ids
);

/* Decode single token ID to UTF-8 piece into out_buf (NUL-terminated). Returns bytes written. */
int tok_decode_piece(
    const IndextTokenizer *tok,
    int token_id,
    char *out_buf,
    size_t out_cap
);

#ifdef __cplusplus
}
#endif

#endif /* INDEXT_TOKENIZER_H */
