package com.indexintegrated.indextloader;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

public class AboutActivity extends Activity {
    public static final String APP_DESCRIPTION =
        "InDEXt Loader runs local language models directly on your device without an internet connection.\n\n" +
        "It loads IDXT models (with an external tokenizer.json) and standard GGUF models (with built-in vocabulary), " +
        "mapping weights directly from storage so they fit within the limited RAM of Android 2.3 - 4.4 devices.\n\n" +
        "Supports both raw text completion and multi-turn chat, with live token streaming and adjustable sampling controls.";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        AppState.applyTheme(this);
        super.onCreate(savedInstanceState);

        int colorPrimary = AppState.getThemeColor(this, android.R.attr.textColorPrimary, 0xFFFFFFFF);
        int colorSecondary = AppState.getThemeColor(this, android.R.attr.textColorSecondary, 0xFFAAAAAA);

        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(16), dp(16), dp(16), dp(20));

        TextView title = new TextView(this);
        title.setText("InDEXt Loader");
        title.setTextSize(22);
        title.setTypeface(null, Typeface.BOLD);
        title.setTextColor(colorPrimary);
        root.addView(title);

        TextView meta = new TextView(this);
        meta.setText("Version 2026.0.0\nDeveloper: InDEX Integrated");
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
        dev.setPadding(0, 0, 0, dp(16));
        root.addView(dev);

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
