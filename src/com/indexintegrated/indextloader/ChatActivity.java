package com.indexintegrated.indextloader;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.os.Handler;
import android.text.ClipboardManager;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

public class ChatActivity extends Activity {
    private static final int MENU_CLEAR          = 1;
    private static final int MENU_MODEL_INFO     = 2;
    private static final int MENU_MODEL_SETTINGS = 3;
    private static final int MENU_APP_SETTINGS   = 4;
    private static final int MENU_UNLOAD         = 5;
    private static final int MENU_EXIT           = 6;

    private LinearLayout msgList;
    private ScrollView scrollLog;
    private EditText etInput;
    private Button btnSend;

    private final Handler handler = new Handler();
    private volatile boolean isGenerating = false;

    private boolean appliedLightTheme;
    private boolean chatMode;
    private boolean addBos;
    private boolean keepKv;
    private boolean autoScroll;
    private boolean showStats;
    private boolean showTimestamps;
    private boolean confirmUnload;
    private String sysPrompt;
    private boolean sysPromptSent = false;
    private String lastStatsRaw = null;
    private float fontSp = 15f;

    private int colorPrimary, colorSecondary;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        appliedLightTheme = AppState.isLightTheme(this);
        AppState.applyTheme(this);
        super.onCreate(savedInstanceState);

        if (AppState.engineHandle == 0) {
            finish();
            return;
        }

        setTitle(AppState.loadedModelName);
        colorPrimary = AppState.getThemeColor(this, android.R.attr.textColorPrimary, 0xFFFFFFFF);
        colorSecondary = AppState.getThemeColor(this, android.R.attr.textColorSecondary, 0xFFAAAAAA);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(10), dp(6), dp(10), dp(8));

        scrollLog = new ScrollView(this);
        msgList = new LinearLayout(this);
        msgList.setOrientation(LinearLayout.VERTICAL);
        msgList.setPadding(dp(2), dp(4), dp(2), dp(4));
        scrollLog.addView(msgList);
        root.addView(scrollLog, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1.0f));

        LinearLayout inputBar = new LinearLayout(this);
        inputBar.setOrientation(LinearLayout.HORIZONTAL);
        inputBar.setGravity(Gravity.BOTTOM);
        inputBar.setPadding(0, dp(6), 0, 0);

        etInput = new EditText(this);
        etInput.setMaxLines(4);
        inputBar.addView(etInput, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));

        btnSend = new Button(this);
        btnSend.setText("Send");
        btnSend.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isGenerating) {
                    NativeEngine.nativeRequestStop(AppState.engineHandle);
                } else {
                    sendPrompt();
                }
            }
        });
        inputBar.addView(btnSend);
        root.addView(inputBar);

        setContentView(root);
        applyLiveSettings();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (AppState.engineHandle == 0) {
            finish();
            return;
        }
        if (appliedLightTheme != AppState.isLightTheme(this)) {
            AppState.recreateActivity(this);
            return;
        }
        applyLiveSettings();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        UiHelper.addOverflowButton(menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == UiHelper.MENU_OVERFLOW_ID) {
            int[] ids = { MENU_CLEAR, MENU_MODEL_INFO, MENU_MODEL_SETTINGS, MENU_APP_SETTINGS, MENU_UNLOAD, MENU_EXIT };
            String[] labels = { "Clear Context", "Model Info", "Model Settings", "App Settings", "Unload Model", "Exit" };
            UiHelper.showOverflowPopup(this, ids, labels, new UiHelper.OverflowAction() {
                @Override
                public void onSelect(int id) {
                    if (id == MENU_CLEAR) clearContext();
                    else if (id == MENU_MODEL_INFO) UiHelper.showModelInfoDialog(ChatActivity.this);
                    else if (id == MENU_MODEL_SETTINGS) startActivity(new Intent(ChatActivity.this, ModelSettingsActivity.class));
                    else if (id == MENU_APP_SETTINGS) startActivity(new Intent(ChatActivity.this, AppSettingsActivity.class));
                    else if (id == MENU_UNLOAD) {
                        if (confirmUnload) {
                            new AlertDialog.Builder(ChatActivity.this)
                                .setTitle("Unload Model")
                                .setMessage("Unload " + AppState.loadedModelName + " from memory?")
                                .setPositiveButton("Unload", new DialogInterface.OnClickListener() {
                                    @Override public void onClick(DialogInterface d, int w) {
                                        if (isGenerating) NativeEngine.nativeRequestStop(AppState.engineHandle);
                                        AppState.releaseEngine();
                                        finish();
                                    }
                                })
                                .setNegativeButton("Cancel", null)
                                .show();
                        } else {
                            if (isGenerating) NativeEngine.nativeRequestStop(AppState.engineHandle);
                            AppState.releaseEngine();
                            finish();
                        }
                    } else if (id == MENU_EXIT) {
                        if (isGenerating) NativeEngine.nativeRequestStop(AppState.engineHandle);
                        finish();
                    }
                }
            });
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private void applyLiveSettings() {
        SharedPreferences p = AppState.prefs(this);
        chatMode = (p.getInt(AppState.KEY_CHAT_MODE, 0) == 1);
        addBos = p.getBoolean(AppState.KEY_ADD_BOS, true);
        keepKv = p.getBoolean(AppState.KEY_KEEP_KV, true);
        autoScroll = p.getBoolean(AppState.KEY_AUTO_SCROLL, true);
        showStats = p.getBoolean(AppState.KEY_SHOW_STATS, true);
        showTimestamps = p.getBoolean(AppState.KEY_TIMESTAMPS, false);
        confirmUnload = p.getBoolean(AppState.KEY_CONFIRM_UNLOAD, false);
        sysPrompt = p.getString(AppState.KEY_SYS_PROMPT, "").trim();

        int fontIdx = p.getInt(AppState.KEY_FONT_SIZE, 1);
        fontSp = (fontIdx == 0) ? 13f : (fontIdx == 2 ? 18f : 15f);
        etInput.setHint(chatMode ? "Message" : "Prompt");
        UiHelper.setActionBarSubtitle(this, formatStatusSub(lastStatsRaw));

        if (p.getBoolean(AppState.KEY_KEEP_SCREEN, false)) {
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        } else {
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        }
    }

    private TextView addMessageItem(String role, String initialText) {
        LinearLayout block = new LinearLayout(this);
        block.setOrientation(LinearLayout.VERTICAL);
        block.setPadding(0, dp(4), 0, dp(8));

        LinearLayout hdr = new LinearLayout(this);
        hdr.setOrientation(LinearLayout.HORIZONTAL);
        hdr.setGravity(Gravity.CENTER_VERTICAL);

        TextView tvRole = new TextView(this);
        if (showTimestamps) {
            String ts = new SimpleDateFormat("HH:mm", Locale.US).format(new Date());
            tvRole.setText(role + " (" + ts + ")");
        } else {
            tvRole.setText(role);
        }
        tvRole.setTextSize(11);
        tvRole.setTypeface(null, Typeface.BOLD);
        tvRole.setTextColor(colorSecondary);
        hdr.addView(tvRole, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));

        final TextView tvBody = new TextView(this);
        tvBody.setText(initialText);
        tvBody.setTextSize(fontSp);
        tvBody.setLineSpacing(dp(2), 1.0f);
        tvBody.setTextColor(colorPrimary);

        Drawable copyIcon = UiHelper.getSystemCopyIcon(this);
        if (copyIcon != null) {
            ImageButton btnCopy = new ImageButton(this);
            btnCopy.setImageDrawable(copyIcon);
            btnCopy.setScaleType(ImageView.ScaleType.FIT_CENTER);
            btnCopy.setBackgroundColor(0x00000000);
            btnCopy.setPadding(dp(3), dp(3), dp(3), dp(3));
            btnCopy.setOnClickListener(new View.OnClickListener() {
                @Override public void onClick(View v) { copyText(tvBody.getText().toString()); }
            });
            hdr.addView(btnCopy, new LinearLayout.LayoutParams(dp(26), dp(26)));
        } else {
            Button btnCopy = new Button(this);
            btnCopy.setText("Copy");
            btnCopy.setTextSize(9);
            btnCopy.setOnClickListener(new View.OnClickListener() {
                @Override public void onClick(View v) { copyText(tvBody.getText().toString()); }
            });
            hdr.addView(btnCopy);
        }

        block.addView(hdr);
        block.addView(tvBody);
        msgList.addView(block);
        return tvBody;
    }

    private String formatStatusSub(String statsRaw) {
        String[] inf = AppState.loadedModelInfo.split("\\|");
        String base = (inf.length >= 10)
            ? (inf[0] + " (" + inf[8] + " MB)")
            : "Ready";
        if (showStats && statsRaw != null && statsRaw.length() > 0) {
            String[] st = statsRaw.split("\\|");
            if (st.length >= 6) return base + " | " + st[5] + " tok/s";
        }
        return base;
    }

    private void sendPrompt() {
        final String text = etInput.getText().toString();
        if (text.trim().length() == 0) return;
        etInput.setText("");

        final String formattedPrompt;
        final TextView tvOut;
        final StringBuilder outBuf = new StringBuilder();

        if (chatMode) {
            String[] inf = AppState.loadedModelInfo.split("\\|");
            boolean isChatML = (inf.length >= 11 && "1".equals(inf[10]));
            StringBuilder pb = new StringBuilder();
            if (isChatML) {
                if (!sysPromptSent && sysPrompt.length() > 0) {
                    pb.append("<|im_start|>system\n").append(sysPrompt).append("<|im_end|>\n");
                    sysPromptSent = true;
                }
                pb.append("<|im_start|>user\n").append(text).append("<|im_end|>\n<|im_start|>assistant\n");
            } else {
                if (!sysPromptSent && sysPrompt.length() > 0) {
                    pb.append(sysPrompt).append("\n");
                    sysPromptSent = true;
                }
                pb.append("<|user|>").append(text).append("<|assistant|>");
            }
            formattedPrompt = pb.toString();
            addMessageItem("You", text);
            tvOut = addMessageItem("Model", "");
        } else {
            formattedPrompt = text;
            addMessageItem("Prompt", text);
            tvOut = addMessageItem("Completion", text);
            outBuf.append(text);
        }
        if (autoScroll) scrollToBottom();

        isGenerating = true;
        btnSend.setText("Stop");

        new Thread(new Runnable() {
            @Override
            public void run() {
                final boolean cont = (chatMode && keepKv);
                int rc = NativeEngine.nativeGenerate(
                    AppState.engineHandle,
                    formattedPrompt,
                    addBos,
                    cont,
                    new NativeEngine.TokenCallback() {
                        @Override
                        public int onToken(byte[] pieceBytes) {
                            String piece;
                            try {
                                piece = new String(pieceBytes, "UTF-8");
                            } catch (Exception e) {
                                piece = new String(pieceBytes);
                            }
                            final String fPiece = piece;
                            handler.post(new Runnable() {
                                @Override
                                public void run() {
                                    outBuf.append(fPiece);
                                    tvOut.setText(outBuf.toString());
                                    if (autoScroll) scrollToBottom();
                                }
                            });
                            return 0;
                        }
                    }
                );
                final String err = (rc != 0) ? NativeEngine.nativeGetLastError() : "";
                final String stats = NativeEngine.nativeGetLastStats(AppState.engineHandle);
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        isGenerating = false;
                        btnSend.setText("Send");
                        lastStatsRaw = stats;
                        UiHelper.setActionBarSubtitle(ChatActivity.this, formatStatusSub(stats));
                        if (err.length() > 0) {
                            Toast.makeText(ChatActivity.this, err, Toast.LENGTH_LONG).show();
                        }
                    }
                });
            }
        }).start();
    }

    private void clearContext() {
        if (isGenerating) NativeEngine.nativeRequestStop(AppState.engineHandle);
        NativeEngine.nativeResetContext(AppState.engineHandle);
        msgList.removeAllViews();
        sysPromptSent = false;
        Toast.makeText(this, "Context cleared", Toast.LENGTH_SHORT).show();
    }

    @SuppressWarnings("deprecation")
    private void copyText(String s) {
        if (s != null && s.length() > 0) {
            ClipboardManager cm = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
            if (cm != null) cm.setText(s);
            Toast.makeText(this, "Copied", Toast.LENGTH_SHORT).show();
        }
    }

    private void scrollToBottom() {
        scrollLog.post(new Runnable() {
            @Override public void run() { scrollLog.fullScroll(View.FOCUS_DOWN); }
        });
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density + 0.5f);
    }
}
