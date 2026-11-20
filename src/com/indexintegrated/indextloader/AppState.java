package com.indexintegrated.indextloader;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.TypedArray;
import android.os.Build;

public final class AppState {
    public static final String PREFS_NAME = "indext_loader_prefs";
    public static final String KEY_WELCOME_DONE = "welcome_done";
    public static final String KEY_THEME_LIGHT  = "theme_light";
    public static final String KEY_FORMAT       = "model_format";
    public static final String KEY_MODEL_PATH   = "model_path";
    public static final String KEY_TOK_PATH     = "tok_path";
    public static final String KEY_THREADS      = "n_threads";
    public static final String KEY_CTX          = "n_ctx";
    public static final String KEY_MAX_TOKENS   = "max_tokens";
    public static final String KEY_TEMP         = "temperature";
    public static final String KEY_TOP_P        = "top_p";
    public static final String KEY_TOP_K        = "top_k";
    public static final String KEY_REPEAT_PEN   = "repeat_penalty";
    public static final String KEY_SEED         = "seed";
    public static final String KEY_CHAT_MODE    = "chat_mode";
    public static final String KEY_SYS_PROMPT   = "sys_prompt";
    public static final String KEY_ADD_BOS      = "add_bos";
    public static final String KEY_KEEP_KV      = "keep_kv";
    public static final String KEY_FONT_SIZE    = "font_size";
    public static final String KEY_AUTO_SCROLL  = "auto_scroll";
    public static final String KEY_SHOW_STATS   = "show_stats";
    public static final String KEY_KEEP_SCREEN  = "keep_screen_on";
    public static final String KEY_TIMESTAMPS   = "show_timestamps";
    public static final String KEY_MEM_WARN     = "mem_warn_toast";
    public static final String KEY_CONFIRM_UNLOAD = "confirm_unload";
    public static final String KEY_DEFAULT_DIR  = "default_dir_mode"; // 0=Download, 1=SDCard, 2=LastUsed
    public static final String KEY_LAST_DIR     = "last_picker_dir";

    public static final int[] THREAD_VALS  = { 0, 1, 2, 4, 6, 8 };
    public static final int[] CTX_VALS     = { 0, 128, 256, 512, 1024, 2048 };
    public static final int[] MAX_TOK_VALS = { 64, 128, 256, 512, 1024 };

    public static long engineHandle = 0;
    public static String loadedModelPath = "";
    public static String loadedTokPath = "";
    public static String loadedModelName = "";
    public static String loadedModelInfo = "";

    public static SharedPreferences prefs(Context ctx) {
        return ctx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }

    public static boolean isLightTheme(Context ctx) {
        return prefs(ctx).getBoolean(KEY_THEME_LIGHT, false);
    }

    public static void applyTheme(Activity act) {
        act.setTheme(isLightTheme(act) ? R.style.AppThemeLight : R.style.AppThemeDark);
    }

    public static int getThemeColor(Context ctx, int attrId, int fallback) {
        TypedArray ta = ctx.obtainStyledAttributes(new int[] { attrId });
        int color = ta.getColor(0, fallback);
        ta.recycle();
        return color;
    }

    public static void recreateActivity(Activity act) {
        if (Build.VERSION.SDK_INT >= 11) {
            act.recreate();
        } else {
            Intent intent = act.getIntent();
            act.finish();
            act.startActivity(intent);
        }
    }

    public static int parseIntSafe(String s, int def) {
        try { return Integer.parseInt(s.trim()); } catch (Exception e) { return def; }
    }

    public static float parseFloatSafe(String s, float def) {
        try { return Float.parseFloat(s.trim().replace(',', '.')); } catch (Exception e) { return def; }
    }

    public static void applyParamsToLiveEngine(Context ctx) {
        if (engineHandle == 0) return;
        SharedPreferences p = prefs(ctx);
        int nThreads = THREAD_VALS[clampIdx(p.getInt(KEY_THREADS, 0), THREAD_VALS.length)];
        int maxTok   = MAX_TOK_VALS[clampIdx(p.getInt(KEY_MAX_TOKENS, 1), MAX_TOK_VALS.length)];
        float temp   = parseFloatSafe(p.getString(KEY_TEMP, "0.7"), 0.7f);
        float topP   = parseFloatSafe(p.getString(KEY_TOP_P, "0.9"), 0.9f);
        int topK     = parseIntSafe(p.getString(KEY_TOP_K, "40"), 40);
        float repPen = parseFloatSafe(p.getString(KEY_REPEAT_PEN, "1.1"), 1.1f);
        int seed     = parseIntSafe(p.getString(KEY_SEED, "0"), 0);
        NativeEngine.nativeUpdateParams(engineHandle, nThreads, maxTok, temp, topP, topK, repPen, seed);
    }

    public static int clampIdx(int idx, int len) {
        if (idx < 0) return 0;
        if (idx >= len) return len - 1;
        return idx;
    }

    public static void releaseEngine() {
        if (engineHandle != 0) {
            NativeEngine.nativeFreeModel(engineHandle);
            engineHandle = 0;
            loadedModelPath = "";
            loadedTokPath = "";
            loadedModelName = "";
            loadedModelInfo = "";
        }
    }
}
