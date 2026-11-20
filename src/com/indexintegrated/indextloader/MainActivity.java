package com.indexintegrated.indextloader;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.util.Locale;

public class MainActivity extends Activity {
    private static final int REQ_MODEL = 101;
    private static final int REQ_TOK = 102;

    private static final int MENU_ABOUT_APP    = 2;
    private static final int MENU_MODEL_INFO   = 3;
    private static final int MENU_APP_SETTINGS = 4;
    private static final int MENU_EXIT         = 5;

    private FrameLayout rootFrame;
    private ScrollView welcomeScroll;
    private LinearLayout loaderContainer;

    private Spinner spFormat;
    private TextView tvModelName, tvModelMeta, tvTokName, tvTokMeta, tvLoadedBanner;
    private LinearLayout tokSection, liveBar;
    private Button btnLaunch, btnRemoveModel, btnRemoveTok;

    private String selectedFormat = "IDXT";
    private String selectedModelPath = "";
    private String selectedTokPath = "";
    private boolean welcomeDone = false;
    private boolean appliedLightTheme;
    private int colorPrimary, colorSecondary;

    private final Handler handler = new Handler();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        appliedLightTheme = AppState.isLightTheme(this);
        AppState.applyTheme(this);
        super.onCreate(savedInstanceState);

        colorPrimary = AppState.getThemeColor(this, android.R.attr.textColorPrimary, 0xFFFFFFFF);
        colorSecondary = AppState.getThemeColor(this, android.R.attr.textColorSecondary, 0xFFAAAAAA);

        SharedPreferences p = AppState.prefs(this);
        welcomeDone = p.getBoolean(AppState.KEY_WELCOME_DONE, false);
        selectedFormat = p.getString(AppState.KEY_FORMAT, "IDXT");
        selectedModelPath = p.getString(AppState.KEY_MODEL_PATH, "");
        selectedTokPath = p.getString(AppState.KEY_TOK_PATH, "");

        UiHelper.setActionBarSubtitle(this, welcomeDone ? "2026.0.0" : null);

        rootFrame = new FrameLayout(this);
        buildWelcomeView();
        buildLoaderView();

        welcomeScroll.setVisibility(welcomeDone ? View.GONE : View.VISIBLE);
        loaderContainer.setVisibility(welcomeDone ? View.VISIBLE : View.GONE);

        rootFrame.addView(welcomeScroll);
        rootFrame.addView(loaderContainer);
        setContentView(rootFrame);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (appliedLightTheme != AppState.isLightTheme(this)) {
            AppState.recreateActivity(this);
            return;
        }
        boolean nowDone = AppState.prefs(this).getBoolean(AppState.KEY_WELCOME_DONE, false);
        if (nowDone != welcomeDone) {
            AppState.recreateActivity(this);
            return;
        }
        updateLiveModelBar();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        if (!welcomeDone) return false;
        UiHelper.addOverflowButton(menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == UiHelper.MENU_OVERFLOW_ID) {
            int[] ids;
            String[] labels;
            if (AppState.engineHandle != 0) {
                ids = new int[] { MENU_ABOUT_APP, MENU_MODEL_INFO, MENU_APP_SETTINGS, MENU_EXIT };
                labels = new String[] { "About App", "Loaded Model Info", "App Settings", "Exit" };
            } else {
                ids = new int[] { MENU_ABOUT_APP, MENU_APP_SETTINGS, MENU_EXIT };
                labels = new String[] { "About App", "App Settings", "Exit" };
            }
            UiHelper.showOverflowPopup(this, ids, labels, new UiHelper.OverflowAction() {
                @Override
                public void onSelect(int itemId) {
                    if (itemId == MENU_ABOUT_APP) startActivity(new Intent(MainActivity.this, AboutActivity.class));
                    else if (itemId == MENU_MODEL_INFO) UiHelper.showModelInfoDialog(MainActivity.this);
                    else if (itemId == MENU_APP_SETTINGS) startActivity(new Intent(MainActivity.this, AppSettingsActivity.class));
                    else if (itemId == MENU_EXIT) { AppState.releaseEngine(); finish(); }
                }
            });
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private void buildWelcomeView() {
        welcomeScroll = new ScrollView(this);
        welcomeScroll.setFillViewport(true);

        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(20), dp(22), dp(20), dp(20));

        TextView title = new TextView(this);
        title.setText("InDEXt Loader");
        title.setTextSize(26);
        title.setTypeface(null, Typeface.BOLD);
        title.setTextColor(colorPrimary);
        title.setPadding(0, 0, 0, dp(14));
        box.addView(title);

        TextView desc = new TextView(this);
        desc.setPadding(0, dp(4), 0, dp(16));
        desc.setTextSize(15);
        desc.setLineSpacing(dp(4), 1.0f);
        desc.setTextColor(colorPrimary);
        desc.setText(AboutActivity.APP_DESCRIPTION + "\n\nTap Start below to choose your model.");
        box.addView(desc);

        View spacer = new View(this);
        box.addView(spacer, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1.0f));

        LinearLayout startBtns = new LinearLayout(this);
        startBtns.setOrientation(LinearLayout.HORIZONTAL);

        Button btnGuide = new Button(this);
        btnGuide.setText("Open Guide");
        btnGuide.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                startActivity(new Intent(MainActivity.this, GuideActivity.class));
            }
        });

        Button btnStart = new Button(this);
        btnStart.setText("Start");
        btnStart.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                welcomeDone = true;
                AppState.prefs(MainActivity.this).edit().putBoolean(AppState.KEY_WELCOME_DONE, true).commit();
                welcomeScroll.setVisibility(View.GONE);
                loaderContainer.setVisibility(View.VISIBLE);
                UiHelper.setActionBarSubtitle(MainActivity.this, "2026.0.0");
                if (Build.VERSION.SDK_INT >= 11) invalidateOptionsMenu();
                Toast.makeText(MainActivity.this, "Select a model file to get started.", Toast.LENGTH_LONG).show();
            }
        });
        startBtns.addView(btnGuide, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));
        startBtns.addView(btnStart, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));
        box.addView(startBtns, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        welcomeScroll.addView(box);
    }

    private void buildLoaderView() {
        loaderContainer = new LinearLayout(this);
        loaderContainer.setOrientation(LinearLayout.VERTICAL);

        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(16), dp(12), dp(16), dp(12));

        liveBar = new LinearLayout(this);
        liveBar.setOrientation(LinearLayout.HORIZONTAL);
        liveBar.setGravity(Gravity.CENTER_VERTICAL);
        liveBar.setPadding(0, 0, 0, dp(10));

        tvLoadedBanner = new TextView(this);
        tvLoadedBanner.setTextSize(14);
        tvLoadedBanner.setTypeface(null, Typeface.BOLD);
        tvLoadedBanner.setTextColor(colorPrimary);
        liveBar.addView(tvLoadedBanner, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));

        Button btnUnload = new Button(this);
        btnUnload.setText("Unload");
        btnUnload.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                AppState.releaseEngine();
                updateLiveModelBar();
                Toast.makeText(MainActivity.this, "Model unloaded", Toast.LENGTH_SHORT).show();
            }
        });
        liveBar.addView(btnUnload);
        root.addView(liveBar);

        root.addView(label("Format"));
        spFormat = new Spinner(this);
        String[] fmts = { "IDXT (.idxt + tokenizer.json)", "GGUF (.gguf)" };
        ArrayAdapter<String> fmtAd = new ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, fmts);
        fmtAd.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spFormat.setAdapter(fmtAd);
        spFormat.setSelection("GGUF".equals(selectedFormat) ? 1 : 0);
        spFormat.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
                selectedFormat = (pos == 1) ? "GGUF" : "IDXT";
                AppState.prefs(MainActivity.this).edit().putString(AppState.KEY_FORMAT, selectedFormat).commit();
                updateFormatVisibility();
                updateLiveModelBar();
            }
            @Override public void onNothingSelected(AdapterView<?> parent) {}
        });
        root.addView(spFormat);

        root.addView(label("Model File"));
        LinearLayout modelRow = makeFileRow();
        tvModelName = (TextView) modelRow.getChildAt(0);
        tvModelMeta = (TextView) modelRow.getChildAt(1);

        Button btnPickModel = new Button(this);
        btnPickModel.setText("Select");
        btnPickModel.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { openModelPicker(); }
        });

        btnRemoveModel = new Button(this);
        btnRemoveModel.setText("Remove");
        btnRemoveModel.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { removeSelectedModel(); }
        });

        LinearLayout modelWrap = new LinearLayout(this);
        modelWrap.setOrientation(LinearLayout.HORIZONTAL);
        modelWrap.setGravity(Gravity.CENTER_VERTICAL);
        modelWrap.addView(modelRow, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));
        modelWrap.addView(btnPickModel);
        modelWrap.addView(btnRemoveModel);
        root.addView(modelWrap);

        tokSection = new LinearLayout(this);
        tokSection.setOrientation(LinearLayout.VERTICAL);
        tokSection.addView(label("Tokenizer File"));
        LinearLayout tokRow = makeFileRow();
        tvTokName = (TextView) tokRow.getChildAt(0);
        tvTokMeta = (TextView) tokRow.getChildAt(1);
        Button btnPickTok = new Button(this);
        btnPickTok.setText("Select");
        btnPickTok.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { openTokPicker(); }
        });
        btnRemoveTok = new Button(this);
        btnRemoveTok.setText("Remove");
        btnRemoveTok.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { removeSelectedTokenizer(); }
        });
        LinearLayout tokWrap = new LinearLayout(this);
        tokWrap.setOrientation(LinearLayout.HORIZONTAL);
        tokWrap.setGravity(Gravity.CENTER_VERTICAL);
        tokWrap.addView(tokRow, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));
        tokWrap.addView(btnPickTok);
        tokWrap.addView(btnRemoveTok);
        tokSection.addView(tokWrap);
        root.addView(tokSection);

        Button btnModelSettings = new Button(this);
        btnModelSettings.setText("Model Settings");
        btnModelSettings.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                startActivity(new Intent(MainActivity.this, ModelSettingsActivity.class));
            }
        });
        LinearLayout.LayoutParams mslp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        mslp.topMargin = dp(12);
        root.addView(btnModelSettings, mslp);

        updateFileLabels();
        updateFormatVisibility();

        scroll.addView(root);
        loaderContainer.addView(scroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1.0f));

        LinearLayout bottom = new LinearLayout(this);
        bottom.setOrientation(LinearLayout.VERTICAL);
        bottom.setPadding(dp(12), dp(6), dp(12), dp(10));
        btnLaunch = new Button(this);
        btnLaunch.setText("Launch");
        btnLaunch.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { launchOrResume(); }
        });
        bottom.addView(btnLaunch, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        loaderContainer.addView(bottom);

        updateLiveModelBar();
    }

    private void removeSelectedModel() {
        if (AppState.engineHandle != 0 && AppState.loadedModelPath.equals(selectedModelPath)) {
            AppState.releaseEngine();
        }
        selectedModelPath = "";
        selectedTokPath = "";
        AppState.prefs(this).edit()
            .remove(AppState.KEY_MODEL_PATH)
            .remove(AppState.KEY_TOK_PATH)
            .commit();
        updateFileLabels();
        updateLiveModelBar();
        Toast.makeText(this, "Model removed from app", Toast.LENGTH_SHORT).show();
    }

    private void removeSelectedTokenizer() {
        if (AppState.engineHandle != 0 && AppState.loadedTokPath.equals(selectedTokPath) && selectedTokPath.length() > 0) {
            AppState.releaseEngine();
        }
        selectedTokPath = "";
        AppState.prefs(this).edit().remove(AppState.KEY_TOK_PATH).commit();
        updateFileLabels();
        updateLiveModelBar();
        Toast.makeText(this, "Tokenizer removed from app", Toast.LENGTH_SHORT).show();
    }

    private LinearLayout makeFileRow() {
        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        col.setPadding(0, dp(4), dp(8), dp(4));
        TextView name = new TextView(this);
        name.setTextSize(15);
        name.setTextColor(colorPrimary);
        col.addView(name);
        TextView meta = new TextView(this);
        meta.setTextSize(12);
        meta.setTextColor(colorSecondary);
        col.addView(meta);
        return col;
    }

    private void updateLiveModelBar() {
        if (AppState.engineHandle != 0) {
            liveBar.setVisibility(View.VISIBLE);
            tvLoadedBanner.setText("Loaded: " + AppState.loadedModelName);
            boolean sameModel = AppState.loadedModelPath.equals(selectedModelPath);
            btnLaunch.setText(sameModel ? "Resume Chat" : "Load & Launch");
        } else {
            liveBar.setVisibility(View.GONE);
            btnLaunch.setText("Launch");
        }
    }

    private void updateFormatVisibility() {
        tokSection.setVisibility("IDXT".equals(selectedFormat) ? View.VISIBLE : View.GONE);
    }

    private void updateFileLabels() {
        if (selectedModelPath.length() == 0) {
            tvModelName.setText("Not selected");
            tvModelMeta.setText("Choose a model file");
            btnRemoveModel.setVisibility(View.GONE);
        } else {
            File f = new File(selectedModelPath);
            tvModelName.setText(f.getName() + " (" + UiHelper.formatFileSize(f.length()) + ")");
            tvModelMeta.setText(f.getParent());
            btnRemoveModel.setVisibility(View.VISIBLE);
        }
        if (selectedTokPath.length() == 0) {
            tvTokName.setText("Not selected");
            tvTokMeta.setText("Required for IDXT");
            btnRemoveTok.setVisibility(View.GONE);
        } else {
            File f = new File(selectedTokPath);
            tvTokName.setText(f.getName() + " (" + UiHelper.formatFileSize(f.length()) + ")");
            tvTokMeta.setText(f.getParent());
            btnRemoveTok.setVisibility(View.VISIBLE);
        }
    }

    private void openModelPicker() {
        Intent i = new Intent(this, FilePickerActivity.class);
        i.putExtra(FilePickerActivity.EXTRA_EXT_FILTER, "GGUF".equals(selectedFormat) ? ".gguf" : ".idxt");
        startActivityForResult(i, REQ_MODEL);
    }

    private void openTokPicker() {
        Intent i = new Intent(this, FilePickerActivity.class);
        i.putExtra(FilePickerActivity.EXTRA_EXT_FILTER, ".json");
        startActivityForResult(i, REQ_TOK);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null) return;
        String path = data.getStringExtra(FilePickerActivity.EXTRA_SELECTED_PATH);
        if (path == null) return;
        if (requestCode == REQ_MODEL) {
            selectedModelPath = path;
            if (path.toLowerCase(Locale.US).endsWith(".gguf")) {
                selectedFormat = "GGUF";
                spFormat.setSelection(1);
            } else if (path.toLowerCase(Locale.US).endsWith(".idxt")) {
                selectedFormat = "IDXT";
                spFormat.setSelection(0);
            }
            AppState.prefs(this).edit()
                .putString(AppState.KEY_FORMAT, selectedFormat)
                .putString(AppState.KEY_MODEL_PATH, selectedModelPath)
                .commit();
            updateFormatVisibility();
            updateFileLabels();
            updateLiveModelBar();
            UiHelper.checkMemoryAndWarn(this, selectedModelPath);
        } else if (requestCode == REQ_TOK) {
            selectedTokPath = path;
            AppState.prefs(this).edit().putString(AppState.KEY_TOK_PATH, selectedTokPath).commit();
            updateFileLabels();
        }
    }

    private void launchOrResume() {
        if (AppState.engineHandle != 0 && AppState.loadedModelPath.equals(selectedModelPath)) {
            AppState.applyParamsToLiveEngine(this);
            startActivity(new Intent(this, ChatActivity.class));
            return;
        }

        if (selectedModelPath.length() == 0 || !new File(selectedModelPath).exists()) {
            Toast.makeText(this, "Select a model file first.", Toast.LENGTH_SHORT).show();
            return;
        }
        final boolean isIdxt = "IDXT".equals(selectedFormat);
        if (isIdxt && (selectedTokPath.length() == 0 || !new File(selectedTokPath).exists())) {
            Toast.makeText(this, "Select tokenizer.json for IDXT.", Toast.LENGTH_SHORT).show();
            return;
        }

        UiHelper.checkMemoryAndWarn(this, selectedModelPath);

        SharedPreferences p = AppState.prefs(this);
        final int nThreads = AppState.THREAD_VALS[AppState.clampIdx(p.getInt(AppState.KEY_THREADS, 0), AppState.THREAD_VALS.length)];
        final int nCtx = AppState.CTX_VALS[AppState.clampIdx(p.getInt(AppState.KEY_CTX, 0), AppState.CTX_VALS.length)];
        final int maxTok = AppState.MAX_TOK_VALS[AppState.clampIdx(p.getInt(AppState.KEY_MAX_TOKENS, 1), AppState.MAX_TOK_VALS.length)];
        final float temp = AppState.parseFloatSafe(p.getString(AppState.KEY_TEMP, "0.7"), 0.7f);
        final float topP = AppState.parseFloatSafe(p.getString(AppState.KEY_TOP_P, "0.9"), 0.9f);
        final int topK = AppState.parseIntSafe(p.getString(AppState.KEY_TOP_K, "40"), 40);
        final float repPen = AppState.parseFloatSafe(p.getString(AppState.KEY_REPEAT_PEN, "1.1"), 1.1f);
        final int seed = AppState.parseIntSafe(p.getString(AppState.KEY_SEED, "0"), 0);
        final String tokArg = (isIdxt && selectedTokPath.length() > 0) ? selectedTokPath : null;

        final ProgressDialog dlg = new ProgressDialog(this);
        dlg.setMessage("Loading model...");
        dlg.setCancelable(false);
        dlg.show();

        new Thread(new Runnable() {
            @Override
            public void run() {
                AppState.releaseEngine();
                final long h = NativeEngine.nativeLoadModel(
                    selectedModelPath, tokArg, nThreads, nCtx, maxTok, temp, topP, topK, repPen, seed
                );
                final String err = (h == 0) ? NativeEngine.nativeGetLastError() : "";
                final String info = (h != 0) ? NativeEngine.nativeGetModelInfo(h) : "";
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        if (dlg.isShowing()) dlg.dismiss();
                        if (h == 0) {
                            new AlertDialog.Builder(MainActivity.this)
                                .setTitle("Load Failed")
                                .setMessage(err.length() > 0 ? err : "Unknown engine error")
                                .setPositiveButton("OK", null)
                                .show();
                        } else {
                            AppState.engineHandle = h;
                            AppState.loadedModelPath = selectedModelPath;
                            AppState.loadedTokPath = tokArg != null ? tokArg : "";
                            AppState.loadedModelName = new File(selectedModelPath).getName();
                            AppState.loadedModelInfo = info;
                            updateLiveModelBar();
                            startActivity(new Intent(MainActivity.this, ChatActivity.class));
                        }
                    }
                });
            }
        }).start();
    }

    private TextView label(String text) {
        TextView tv = new TextView(this);
        tv.setText(text);
        tv.setTextSize(12);
        tv.setTypeface(null, Typeface.BOLD);
        tv.setTextColor(colorSecondary);
        tv.setPadding(0, dp(10), 0, dp(2));
        return tv;
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density + 0.5f);
    }
}
