package com.indexintegrated.indextloader;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.Toast;

public class AppSettingsActivity extends Activity {
    private Spinner spFontSize, spDefaultDir;
    private CheckBox cbAutoScroll, cbShowStats, cbTimestamps, cbKeepScreen, cbMemWarn, cbLoadDiag, cbConfirmUnload, cbLightTheme;

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

        root.addView(UiHelper.sectionHeader(this, "APPEARANCE", colorSecondary));
        spFontSize = UiHelper.makeSpinnerRow(this, root, "Output Font Size", new String[] { "Small (13sp)", "Medium (15sp)", "Large (18sp)" }, p.getInt(AppState.KEY_FONT_SIZE, 1), colorPrimary);
        cbLightTheme = UiHelper.makeCheck(this, root, Build.VERSION.SDK_INT >= 11 ? "Holo Light theme" : "Light theme", AppState.isLightTheme(this));
        cbLightTheme.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                saveAll();
                AppState.prefs(AppSettingsActivity.this).edit().putBoolean(AppState.KEY_THEME_LIGHT, isChecked).commit();
                AppState.recreateActivity(AppSettingsActivity.this);
            }
        });

        root.addView(UiHelper.sectionHeader(this, "CHAT & DISPLAY", colorSecondary));
        cbAutoScroll = UiHelper.makeCheck(this, root, "Auto-scroll output", p.getBoolean(AppState.KEY_AUTO_SCROLL, true));
        cbShowStats = UiHelper.makeCheck(this, root, "Show generation speed (tok/s)", p.getBoolean(AppState.KEY_SHOW_STATS, true));
        cbTimestamps = UiHelper.makeCheck(this, root, "Show message timestamps", p.getBoolean(AppState.KEY_TIMESTAMPS, false));
        cbKeepScreen = UiHelper.makeCheck(this, root, "Keep screen on during chat", p.getBoolean(AppState.KEY_KEEP_SCREEN, false));

        root.addView(UiHelper.sectionHeader(this, "STORAGE & DIAGNOSTICS", colorSecondary));
        spDefaultDir = UiHelper.makeSpinnerRow(this, root, "File Browser Start", new String[] { "Download", "SDCard Root", "Last Used" }, p.getInt(AppState.KEY_DEFAULT_DIR, 0), colorPrimary);
        cbMemWarn = UiHelper.makeCheck(this, root, "Low memory warning toast", p.getBoolean(AppState.KEY_MEM_WARN, true));
        cbLoadDiag = UiHelper.makeCheck(this, root, "Show load diagnostics toast", p.getBoolean(AppState.KEY_LOAD_DIAG, true));
        cbConfirmUnload = UiHelper.makeCheck(this, root, "Confirm before unloading model", p.getBoolean(AppState.KEY_CONFIRM_UNLOAD, false));

        root.addView(UiHelper.sectionHeader(this, "DEFAULTS", colorSecondary));
        Button btnReset = new Button(this);
        btnReset.setText("Reset All Settings");
        btnReset.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                new AlertDialog.Builder(AppSettingsActivity.this)
                    .setTitle("Reset Settings")
                    .setMessage("Reset all app and model settings to defaults?")
                    .setPositiveButton("Reset", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            boolean w = AppState.prefs(AppSettingsActivity.this).getBoolean(AppState.KEY_WELCOME_DONE, true);
                            AppState.prefs(AppSettingsActivity.this).edit().clear().putBoolean(AppState.KEY_WELCOME_DONE, w).commit();
                            Toast.makeText(AppSettingsActivity.this, "Settings reset", Toast.LENGTH_SHORT).show();
                            AppState.recreateActivity(AppSettingsActivity.this);
                        }
                    })
                    .setNegativeButton("Cancel", null)
                    .show();
            }
        });
        root.addView(btnReset, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        scroll.addView(root);
        outer.addView(scroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1.0f));

        LinearLayout bottom = new LinearLayout(this);
        bottom.setOrientation(LinearLayout.VERTICAL);
        bottom.setPadding(UiHelper.dp(this, 12), UiHelper.dp(this, 6), UiHelper.dp(this, 12), UiHelper.dp(this, 10));
        Button btnDone = new Button(this);
        btnDone.setText("Save & Back");
        btnDone.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { saveAll(); finish(); }
        });
        bottom.addView(btnDone, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        outer.addView(bottom);

        setContentView(outer);
    }

    @Override
    protected void onPause() {
        super.onPause();
        saveAll();
    }

    private void saveAll() {
        AppState.prefs(this).edit()
            .putInt(AppState.KEY_FONT_SIZE, spFontSize.getSelectedItemPosition())
            .putInt(AppState.KEY_DEFAULT_DIR, spDefaultDir.getSelectedItemPosition())
            .putBoolean(AppState.KEY_AUTO_SCROLL, cbAutoScroll.isChecked())
            .putBoolean(AppState.KEY_SHOW_STATS, cbShowStats.isChecked())
            .putBoolean(AppState.KEY_TIMESTAMPS, cbTimestamps.isChecked())
            .putBoolean(AppState.KEY_KEEP_SCREEN, cbKeepScreen.isChecked())
            .putBoolean(AppState.KEY_MEM_WARN, cbMemWarn.isChecked())
            .putBoolean(AppState.KEY_LOAD_DIAG, cbLoadDiag.isChecked())
            .putBoolean(AppState.KEY_CONFIRM_UNLOAD, cbConfirmUnload.isChecked())
            .commit();
    }
}
