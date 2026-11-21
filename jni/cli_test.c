#include "indext_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int on_token(const char *piece, int token_id, void *user_data) {
    (void)token_id;
    (void)user_data;
    fputs(piece, stdout);
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <model.idxt|model.gguf> [tokenizer.json] [prompt] [add_bos] [use_mmap] [use_fp16_kv]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *tok_path = (argc >= 3 && argv[2][0] != '-') ? argv[2] : NULL;
    if (tok_path && strcmp(tok_path, "none") == 0) tok_path = NULL;
    const char *prompt = (argc >= 4) ? argv[3] : "Once upon a time";
    int add_bos = (argc >= 5) ? atoi(argv[4]) : 1;

    IndextParams p = indext_default_params();
    p.temperature = 0.0f; /* greedy for deterministic verification */
    p.max_tokens = 48;
    if (argc >= 6) p.use_mmap = atoi(argv[5]);
    if (argc >= 7) p.kv_mode = atoi(argv[6]);

    char err[512] = {0};
    IndextContext *ctx = indext_load(model_path, tok_path, &p, err, sizeof(err));
    if (!ctx) {
        fprintf(stderr, "LOAD ERROR: %s\n", err);
        return 1;
    }

    IndextModelInfo info;
    indext_get_info(ctx, &info);
    printf("[Loaded] fmt=%d v=%d arch=%s dim=%u layers=%u heads=%u kv_heads=%u vocab=%u ctx=%u mmap=%d kv_mode=%d kv_mb=%.2f\n",
           info.format, info.idxt_version, info.arch, info.dim, info.n_layers, info.n_heads, info.n_kv_heads,
           info.vocab_size, info.max_seq_len, info.used_mmap, info.kv_mode,
           (double)info.kv_cache_bytes / (1024.0 * 1024.0));

    printf("Prompt: %s", prompt);
    fflush(stdout);

    if (indext_generate(ctx, prompt, add_bos, 0, on_token, NULL, err, sizeof(err)) != 0) {
        fprintf(stderr, "\nGEN ERROR: %s\n", err);
        indext_free(ctx);
        return 1;
    }

    IndextStats st;
    indext_get_stats(ctx, &st);
    printf("\n[Stats] load=%.1fms prompt=%d(%.1fms) gen=%d(%.1fms, %.2f tok/s)\n",
           st.load_time_ms, st.prompt_tokens, st.prompt_eval_ms, st.tokens_generated, st.gen_time_ms, st.tok_per_sec);

    indext_free(ctx);
    return 0;
}
