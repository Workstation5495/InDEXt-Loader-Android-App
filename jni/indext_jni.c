#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "indext_engine.h"

static char g_last_error[512] = {0};

typedef struct {
    JNIEnv *env;
    jobject cb_obj;
    jmethodID on_token_mid;
} JniCbCtx;

static int jni_token_cb(const char *piece, int token_id, void *user_data) {
    (void)token_id;
    JniCbCtx *jc = (JniCbCtx *)user_data;
    if (!jc || !jc->cb_obj || !jc->on_token_mid) return 0;

    int len = piece ? (int)strlen(piece) : 0;
    jbyteArray jbytes = (*jc->env)->NewByteArray(jc->env, len);
    if (len > 0) {
        (*jc->env)->SetByteArrayRegion(jc->env, jbytes, 0, len, (const jbyte *)piece);
    }
    jint stop = (*jc->env)->CallIntMethod(jc->env, jc->cb_obj, jc->on_token_mid, jbytes);
    (*jc->env)->DeleteLocalRef(jc->env, jbytes);
    if ((*jc->env)->ExceptionCheck(jc->env)) {
        (*jc->env)->ExceptionClear(jc->env);
        return 1;
    }
    return stop != 0 ? 1 : 0;
}

JNIEXPORT jstring JNICALL
Java_com_indexintegrated_indextloader_NativeEngine_nativeGetLastError(JNIEnv *env, jclass cls) {
    (void)cls;
    return (*env)->NewStringUTF(env, g_last_error);
}

JNIEXPORT jlong JNICALL
Java_com_indexintegrated_indextloader_NativeEngine_nativeLoadModel(
    JNIEnv *env, jclass cls,
    jstring jmodel, jstring jtok,
    jint n_threads, jint n_ctx, jint max_tokens,
    jfloat temp, jfloat top_p, jint top_k, jfloat repeat_penalty, jint seed,
    jboolean use_mmap, jint kv_mode
) {
    (void)cls;
    g_last_error[0] = '\0';
    const char *model_path = jmodel ? (*env)->GetStringUTFChars(env, jmodel, NULL) : NULL;
    const char *tok_path   = jtok   ? (*env)->GetStringUTFChars(env, jtok, NULL)   : NULL;

    IndextParams p = indext_default_params();
    p.n_threads = (int)n_threads;
    p.n_ctx = (int)n_ctx;
    p.max_tokens = (int)max_tokens;
    p.temperature = (float)temp;
    p.top_p = (float)top_p;
    p.top_k = (int)top_k;
    p.repeat_penalty = (float)repeat_penalty;
    p.seed = (int)seed;
    p.use_mmap = use_mmap ? 1 : 0;
    p.kv_mode = (int)kv_mode;

    IndextContext *ctx = indext_load(model_path, tok_path, &p, g_last_error, sizeof(g_last_error));

    if (model_path) (*env)->ReleaseStringUTFChars(env, jmodel, model_path);
    if (tok_path)   (*env)->ReleaseStringUTFChars(env, jtok, tok_path);
    return (jlong)(intptr_t)ctx;
}

JNIEXPORT void JNICALL
Java_com_indexintegrated_indextloader_NativeEngine_nativeFreeModel(JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    IndextContext *ctx = (IndextContext *)(intptr_t)handle;
    if (ctx) indext_free(ctx);
}

JNIEXPORT void JNICALL
Java_com_indexintegrated_indextloader_NativeEngine_nativeResetContext(JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    IndextContext *ctx = (IndextContext *)(intptr_t)handle;
    if (ctx) indext_reset_kv(ctx);
}

JNIEXPORT void JNICALL
Java_com_indexintegrated_indextloader_NativeEngine_nativeRequestStop(JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    IndextContext *ctx = (IndextContext *)(intptr_t)handle;
    if (ctx) indext_request_stop(ctx);
}

JNIEXPORT void JNICALL
Java_com_indexintegrated_indextloader_NativeEngine_nativeUpdateParams(
    JNIEnv *env, jclass cls, jlong handle,
    jint n_threads, jint max_tokens, jfloat temp, jfloat top_p, jint top_k, jfloat repeat_penalty, jint seed
) {
    (void)env; (void)cls;
    IndextContext *ctx = (IndextContext *)(intptr_t)handle;
    if (!ctx) return;
    IndextParams p = indext_default_params();
    p.n_threads = (int)n_threads;
    p.max_tokens = (int)max_tokens;
    p.temperature = (float)temp;
    p.top_p = (float)top_p;
    p.top_k = (int)top_k;
    p.repeat_penalty = (float)repeat_penalty;
    p.seed = (int)seed;
    indext_update_params(ctx, &p);
}

JNIEXPORT jstring JNICALL
Java_com_indexintegrated_indextloader_NativeEngine_nativeGetModelInfo(JNIEnv *env, jclass cls, jlong handle) {
    (void)cls;
    IndextContext *ctx = (IndextContext *)(intptr_t)handle;
    if (!ctx) return (*env)->NewStringUTF(env, "");
    IndextModelInfo info;
    indext_get_info(ctx, &info);
    const char *fmt_str = (info.format == INDEXT_FMT_IDXT) ? "IDXT" : "GGUF";
    char buf[512];
    snprintf(buf, sizeof(buf), "%s|%s|%u|%u|%u|%u|%u|%u|%.2f|%.2f|%d|%d|%d",
             fmt_str,
             info.arch,
             info.vocab_size,
             info.dim,
             info.n_layers,
             info.n_heads,
             info.n_kv_heads,
             info.max_seq_len,
             (double)info.file_size / (1024.0 * 1024.0),
             (double)info.kv_cache_bytes / (1024.0 * 1024.0),
             info.is_chatml,
             info.used_mmap,
             info.kv_mode);
    return (*env)->NewStringUTF(env, buf);
}

JNIEXPORT jstring JNICALL
Java_com_indexintegrated_indextloader_NativeEngine_nativeGetLastStats(JNIEnv *env, jclass cls, jlong handle) {
    (void)cls;
    IndextContext *ctx = (IndextContext *)(intptr_t)handle;
    if (!ctx) return (*env)->NewStringUTF(env, "");
    IndextStats st;
    indext_get_stats(ctx, &st);
    char buf[256];
    snprintf(buf, sizeof(buf), "%d|%d|%.0f|%.0f|%.0f|%.2f",
             st.tokens_generated, st.prompt_tokens,
             st.load_time_ms, st.prompt_eval_ms, st.gen_time_ms, st.tok_per_sec);
    return (*env)->NewStringUTF(env, buf);
}

JNIEXPORT jint JNICALL
Java_com_indexintegrated_indextloader_NativeEngine_nativeGenerate(
    JNIEnv *env, jclass cls, jlong handle,
    jstring jprompt, jboolean add_bos, jboolean continue_ctx, jobject callback
) {
    (void)cls;
    IndextContext *ctx = (IndextContext *)(intptr_t)handle;
    if (!ctx || !jprompt) return -1;

    const char *prompt = (*env)->GetStringUTFChars(env, jprompt, NULL);
    JniCbCtx jc = {0};
    jc.env = env;
    jc.cb_obj = callback;
    if (callback) {
        jclass cb_cls = (*env)->GetObjectClass(env, callback);
        jc.on_token_mid = (*env)->GetMethodID(env, cb_cls, "onToken", "([B)I");
        (*env)->DeleteLocalRef(env, cb_cls);
    }

    int rc = indext_generate(ctx, prompt, add_bos ? 1 : 0, continue_ctx ? 1 : 0,
                             jni_token_cb, &jc, g_last_error, sizeof(g_last_error));
    (*env)->ReleaseStringUTFChars(env, jprompt, prompt);
    return (jint)rc;
}
