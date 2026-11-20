package com.indexintegrated.indextloader;

import android.app.Activity;
import android.graphics.Typeface;
import android.os.Bundle;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

public class GuideActivity extends Activity {
    private int colorPrimary, colorSecondary;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        AppState.applyTheme(this);
        super.onCreate(savedInstanceState);

        colorPrimary = AppState.getThemeColor(this, android.R.attr.textColorPrimary, 0xFFFFFFFF);
        colorSecondary = AppState.getThemeColor(this, android.R.attr.textColorSecondary, 0xFFAAAAAA);

        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(16), dp(14), dp(16), dp(20));

        addSection(root, "Getting Started",
            "Choose a model format on the main screen, select your model file from storage, and tap Launch. Once loaded, the model stays in memory while you switch between chat and settings.");

        addSection(root, "Supported Formats",
            "GGUF: Self-contained single file with weights and vocabulary inside. Supports F32, F16, Q8_0, Q4_0, Q4_1, Q5_0, Q4_K, and Q6_K.\n\n" +
            "IDXT: Compact Q8_0 format. Requires selecting both the .idxt model file and its matching tokenizer.json.");

        addSection(root, "Prompt Modes (Model Settings)",
            "Completion: Continues your text directly. Best for base and pretrain models.\n\n" +
            "Chat: Formats turns with <|user|> and <|assistant|> tags and keeps conversation history in the KV cache.");

        addSection(root, "Tips for Older Devices",
            "Lower Context Length if you hit memory limits.\n" +
            "Set CPU Threads to 2 or 4 if Auto feels slower.\n" +
            "Use the copy icon next to any message in chat to copy its text.");

        scroll.addView(root, new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        setContentView(scroll);
    }

    private void addSection(LinearLayout parent, String title, String body) {
        TextView tvTitle = new TextView(this);
        tvTitle.setText(title);
        tvTitle.setTextSize(15);
        tvTitle.setTypeface(null, Typeface.BOLD);
        tvTitle.setTextColor(colorSecondary);
        tvTitle.setPadding(0, dp(10), 0, dp(4));
        parent.addView(tvTitle);

        TextView tvBody = new TextView(this);
        tvBody.setText(body);
        tvBody.setTextSize(14);
        tvBody.setLineSpacing(dp(3), 1.0f);
        tvBody.setTextColor(colorPrimary);
        tvBody.setPadding(0, 0, 0, dp(8));
        parent.addView(tvBody);
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density + 0.5f);
    }
}
