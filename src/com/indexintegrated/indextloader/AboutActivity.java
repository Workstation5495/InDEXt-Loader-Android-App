package com.indexintegrated.indextloader;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

public class AboutActivity extends Activity {
    public static final String GITHUB_URL = "https://github.com/Workstation5495/InDEXt-Loader-Android-App/";
    public static final String HF_URL = "https://huggingface.co/ResonateIntegrated/InDEXt-25M";

    public static final String APP_DESCRIPTION =
        "InDEXt Loader runs InDEX Transformer and other local language models directly on your device without an internet connection, powered by the InDEX-Transformer.C engine.\n\n" +
        "It loads IDXT models (with an external tokenizer.json) and standard GGUF models (with built-in vocabulary), " +
        "using ARM NEON SIMD acceleration, all GGML quantizations (F32 to Q2_K), 4 KV cache modes (FP32/FP16/Q8_0/Q4_0), " +
        "and optional mmap zero-copy loading for Android 2.3+ and modern devices.\n\n" +
        "Supports LLaMA, Qwen 2/2.5/3, Gemma 1/2/3, Phi-3, SmolLM2, Mistral, StarCoder2, GPT-2, multi-turn chat templates, and TXT/JSON export.";

    public static void openUrl(Activity act, String url) {
        try {
            Intent i = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
            act.startActivity(i);
        } catch (Exception e) {
            Toast.makeText(act, "No browser found: " + url, Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        AppState.applyTheme(this);
        super.onCreate(savedInstanceState);

        int colorPrimary = AppState.getThemeColor(this, android.R.attr.textColorPrimary, 0xFFFFFFFF);
        int colorSecondary = AppState.getThemeColor(this, android.R.attr.textColorSecondary, 0xFFAAAAAA);

        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(16), dp(16), dp(16), dp(16));

        TextView title = new TextView(this);
        title.setText("InDEXt Loader");
        title.setTextSize(22);
        title.setTypeface(null, Typeface.BOLD);
        title.setTextColor(colorPrimary);
        root.addView(title);

        TextView meta = new TextView(this);
        meta.setText("Version 26.0.1\nDeveloper: InDEX Integrated");
        meta.setTextSize(13);
        meta.setTextColor(colorSecondary);
        meta.setPadding(0, dp(4), 0, dp(14));
        root.addView(meta);

        TextView desc = new TextView(this);
        desc.setText(APP_DESCRIPTION);
        desc.setTextSize(14);
        desc.setLineSpacing(dp(3), 1.0f);
        desc.setTextColor(colorPrimary);
        desc.setPadding(0, 0, 0, dp(16));
        root.addView(desc);

        Runtime rt = Runtime.getRuntime();
        TextView dev = new TextView(this);
        dev.setText(
            "Device: " + Build.MODEL + "\n" +
            "Android: " + Build.VERSION.RELEASE + " (API " + Build.VERSION.SDK_INT + ")\n" +
            "CPU ABI: " + Build.CPU_ABI + " (" + rt.availableProcessors() + " cores)"
        );
        dev.setTextSize(13);
        dev.setTextColor(colorSecondary);
        dev.setPadding(0, 0, 0, dp(14));
        root.addView(dev);

        LinearLayout linkRow = new LinearLayout(this);
        linkRow.setOrientation(LinearLayout.HORIZONTAL);
        Button btnGh = new Button(this);
        btnGh.setText("GitHub");
        btnGh.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { openUrl(AboutActivity.this, GITHUB_URL); }
        });
        Button btnHf = new Button(this);
        btnHf.setText("HuggingFace");
        btnHf.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { openUrl(AboutActivity.this, HF_URL); }
        });
        linkRow.addView(btnGh, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));
        linkRow.addView(btnHf, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));
        root.addView(linkRow, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        View spacer = new View(this);
        root.addView(spacer, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1.0f));

        Button btnGuide = new Button(this);
        btnGuide.setText("Open Guide");
        btnGuide.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                startActivity(new Intent(AboutActivity.this, GuideActivity.class));
            }
        });
        root.addView(btnGuide, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        scroll.addView(root, new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        setContentView(scroll);
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density + 0.5f);
    }
}
