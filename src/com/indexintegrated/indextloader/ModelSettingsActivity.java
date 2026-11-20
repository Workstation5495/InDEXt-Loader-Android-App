package com.indexintegrated.indextloader;

import android.app.Activity;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

public class ModelSettingsActivity extends Activity {
    private Spinner spThreads, spCtx, spMaxTok, spMode;
    private EditText etTemp, etTopP, etTopK, etRepPen, etSeed, etSysPrompt;
    private CheckBox cbAddBos, cbKeepKv;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        AppState.applyTheme(this);
        super.onCreate(savedInstanceState);

        int colorPrimary = AppState.getThemeColor(this, android.R.attr.textColorPrimary, 0xFFFFFFFF);
        int colorSecondary = AppState.getThemeColor(this, android.R.attr.textColorSecondary, 0xFFAAAAAA);

        SharedPreferences p = AppState.prefs(this);

        LinearLayout outer = new LinearLayout(this);
        outer.setOrientation(LinearLayout.VERTICAL);

        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(UiHelper.dp(this, 16), UiHelper.dp(this, 12), UiHelper.dp(this, 16), UiHelper.dp(this, 14));

        root.addView(UiHelper.sectionHeader(this, "HARDWARE & CONTEXT", colorSecondary));
        spThreads = UiHelper.makeSpinnerRow(this, root, "CPU Threads", new String[] { "Auto", "1", "2", "4", "6", "8" }, p.getInt(AppState.KEY_THREADS, 0), colorPrimary);
        spCtx = UiHelper.makeSpinnerRow(this, root, "Context Length (on load)", new String[] { "Model Default", "128", "256", "512", "1024", "2048" }, p.getInt(AppState.KEY_CTX, 0), colorPrimary);

        cbAddBos = UiHelper.makeCheck(this, root, "Prepend BOS token", p.getBoolean(AppState.KEY_ADD_BOS, true));
        cbKeepKv = UiHelper.makeCheck(this, root, "Keep KV cache between turns", p.getBoolean(AppState.KEY_KEEP_KV, true));

        root.addView(UiHelper.sectionHeader(this, "GENERATION & SAMPLING", colorSecondary));
        spMode = UiHelper.makeSpinnerRow(this, root, "Prompt Mode", new String[] { "Completion", "Chat (<|user|> / <|assistant|>)" }, p.getInt(AppState.KEY_CHAT_MODE, 0), colorPrimary);
        spMaxTok = UiHelper.makeSpinnerRow(this, root, "Max New Tokens", new String[] { "64", "128", "256", "512", "1024" }, p.getInt(AppState.KEY_MAX_TOKENS, 1), colorPrimary);
        etTemp = UiHelper.makeEditRow(this, root, "Temperature (0 = greedy)", p.getString(AppState.KEY_TEMP, "0.7"), colorPrimary);
        etTopP = UiHelper.makeEditRow(this, root, "Top-P", p.getString(AppState.KEY_TOP_P, "0.9"), colorPrimary);
        etTopK = UiHelper.makeEditRow(this, root, "Top-K", p.getString(AppState.KEY_TOP_K, "40"), colorPrimary);
        etRepPen = UiHelper.makeEditRow(this, root, "Repeat Penalty", p.getString(AppState.KEY_REPEAT_PEN, "1.1"), colorPrimary);
        etSeed = UiHelper.makeEditRow(this, root, "Seed (0 = random)", p.getString(AppState.KEY_SEED, "0"), colorPrimary);

        TextView sysLbl = new TextView(this);
        sysLbl.setText("System Prompt (Chat mode)");
        sysLbl.setTextSize(13);
        sysLbl.setTextColor(colorPrimary);
        sysLbl.setPadding(0, UiHelper.dp(this, 8), 0, UiHelper.dp(this, 2));
        root.addView(sysLbl);

        etSysPrompt = new EditText(this);
        etSysPrompt.setHint("Optional");
        etSysPrompt.setText(p.getString(AppState.KEY_SYS_PROMPT, ""));
        etSysPrompt.setMaxLines(3);
        root.addView(etSysPrompt);

        scroll.addView(root);
        outer.addView(scroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1.0f));

        LinearLayout bottom = new LinearLayout(this);
        bottom.setOrientation(LinearLayout.VERTICAL);
        bottom.setPadding(UiHelper.dp(this, 12), UiHelper.dp(this, 6), UiHelper.dp(this, 12), UiHelper.dp(this, 10));
        Button btnApply = new Button(this);
        btnApply.setText("Apply & Back");
        btnApply.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                saveAll();
                AppState.applyParamsToLiveEngine(ModelSettingsActivity.this);
                Toast.makeText(ModelSettingsActivity.this, "Model settings applied", Toast.LENGTH_SHORT).show();
                finish();
            }
        });
        bottom.addView(btnApply, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        outer.addView(bottom);

        setContentView(outer);
    }

    @Override
    protected void onPause() {
        super.onPause();
        saveAll();
        AppState.applyParamsToLiveEngine(this);
    }

    private void saveAll() {
        AppState.prefs(this).edit()
            .putInt(AppState.KEY_THREADS, spThreads.getSelectedItemPosition())
            .putInt(AppState.KEY_CTX, spCtx.getSelectedItemPosition())
            .putInt(AppState.KEY_MAX_TOKENS, spMaxTok.getSelectedItemPosition())
            .putInt(AppState.KEY_CHAT_MODE, spMode.getSelectedItemPosition())
            .putString(AppState.KEY_TEMP, etTemp.getText().toString().trim())
            .putString(AppState.KEY_TOP_P, etTopP.getText().toString().trim())
            .putString(AppState.KEY_TOP_K, etTopK.getText().toString().trim())
            .putString(AppState.KEY_REPEAT_PEN, etRepPen.getText().toString().trim())
            .putString(AppState.KEY_SEED, etSeed.getText().toString().trim())
            .putString(AppState.KEY_SYS_PROMPT, etSysPrompt.getText().toString())
            .putBoolean(AppState.KEY_ADD_BOS, cbAddBos.isChecked())
            .putBoolean(AppState.KEY_KEEP_KV, cbKeepKv.isChecked())
            .commit();
    }
}
