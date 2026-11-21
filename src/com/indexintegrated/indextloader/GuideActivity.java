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

        addSection(root, "Getting Started (InDEX-Transformer.C)",
            "Choose a model format on the main screen, select your model file from storage, and tap Launch. Once loaded, the model stays in memory while you switch between chat and settings.");

        addSection(root, "Supported Formats & Architectures",
            "GGUF: Self-contained single file with weights and vocabulary inside. Supports F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K.\n\n" +
            "IDXT: Compact native format supporting all quantizations above. Requires selecting both the .idxt model file and its matching tokenizer.json.\n\n" +
            "Architectures: LLaMA, Mistral, SmolLM2, Qwen 2/2.5/3, Gemma 1/2, Phi-3/3.5, StarCoder2, StableLM, and GPT-2.");

        addSection(root, "Prompt & Chat Templates (Model Settings)",
            "Completion: Continues raw text directly.\n" +
            "Auto Chat: Detects ChatML from GGUF or falls back to <|user|>/<|assistant|>.\n" +
            "ChatML, LLaMA 3, Gemma, Alpaca, Phi/Zephyr, Mistral: Explicit chat formats you can pick for any model.");

        addSection(root, "Performance & Export",
            "Use Q8_0 (4x less RAM) or Q4_0 (8x less RAM) KV Cache for maximum speed and context length on older phones.\n" +
            "Use mmap (zero-copy) for large models, or disable mmap in Model Settings if your device has enough RAM and flash I/O stalls.\n" +
            "In Chat, open the menu and tap Export Chat to save your conversation as .txt or .json to any folder.");

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
