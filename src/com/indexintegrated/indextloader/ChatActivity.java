package com.indexintegrated.indextloader;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.text.ClipboardManager;
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

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class ChatActivity extends Activity {
    private static final int REQ_EXPORT_DIR = 201;

    private static final int MENU_CLEAR          = 1;
    private static final int MENU_EXPORT         = 2;
    private static final int MENU_MODEL_INFO     = 3;
    private static final int MENU_MODEL_SETTINGS = 4;
    private static final int MENU_APP_SETTINGS   = 5;
    private static final int MENU_UNLOAD         = 6;
    private static final int MENU_EXIT           = 7;

    private static final class ChatEntry {
        String role;
        String text;
        String timestamp;
    }

    private LinearLayout msgList;
    private ScrollView scrollLog;
    private EditText etInput;
    private Button btnSend;

    private final List<ChatEntry> chatHistory = new ArrayList<ChatEntry>();
    private boolean pendingExportJson = false;

    private final Handler handler = new Handler();
    private volatile boolean isGenerating = false;

    private boolean appliedLightTheme;
    private int chatModeIdx;
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
            int[] ids = { MENU_CLEAR, MENU_EXPORT, MENU_MODEL_INFO, MENU_MODEL_SETTINGS, MENU_APP_SETTINGS, MENU_UNLOAD, MENU_EXIT };
            String[] labels = { "Clear Context", "Export Chat", "Model Info", "Model Settings", "App Settings", "Unload Model", "Exit" };
            UiHelper.showOverflowPopup(this, ids, labels, new UiHelper.OverflowAction() {
                @Override
                public void onSelect(int id) {
                    if (id == MENU_CLEAR) clearContext();
                    else if (id == MENU_EXPORT) showExportDialog();
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
        chatModeIdx = p.getInt(AppState.KEY_CHAT_MODE, 0);
        chatMode = (chatModeIdx != 0);
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

    private TextView addMessageItem(String role, String initialText, ChatEntry entryOut) {
        String ts = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
        entryOut.role = role;
        entryOut.text = initialText;
        entryOut.timestamp = ts;
        chatHistory.add(entryOut);

        LinearLayout block = new LinearLayout(this);
        block.setOrientation(LinearLayout.VERTICAL);
        block.setPadding(0, dp(4), 0, dp(8));

        LinearLayout hdr = new LinearLayout(this);
        hdr.setOrientation(LinearLayout.HORIZONTAL);
        hdr.setGravity(Gravity.CENTER_VERTICAL);

        TextView tvRole = new TextView(this);
        if (showTimestamps) {
            tvRole.setText(role + " (" + ts.substring(0, 5) + ")");
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
        final ChatEntry outEntry = new ChatEntry();

        if (chatMode) {
            String[] inf = AppState.loadedModelInfo.split("\\|");
            boolean isChatML = (inf.length >= 11 && "1".equals(inf[10]));
            int mode = (chatModeIdx == 1) ? (isChatML ? 2 : 1) : chatModeIdx;
            StringBuilder pb = new StringBuilder();
            if (mode == 2) {
                if (!sysPromptSent && sysPrompt.length() > 0) {
                    pb.append("<|im_start|>system\n").append(sysPrompt).append("<|im_end|>\n");
                    sysPromptSent = true;
                }
                pb.append("<|im_start|>user\n").append(text).append("<|im_end|>\n<|im_start|>assistant\n");
            } else if (mode == 3) {
                if (!sysPromptSent && sysPrompt.length() > 0) {
                    pb.append("<|start_header_id|>system<|end_header_id|>\n\n").append(sysPrompt).append("<|eot_id|>");
                    sysPromptSent = true;
                }
                pb.append("<|start_header_id|>user<|end_header_id|>\n\n").append(text).append("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
            } else if (mode == 4) {
                if (!sysPromptSent && sysPrompt.length() > 0) {
                    pb.append("<start_of_turn>user\n").append(sysPrompt).append("\n\n").append(text).append("<end_of_turn>\n<start_of_turn>model\n");
                    sysPromptSent = true;
                } else {
                    pb.append("<start_of_turn>user\n").append(text).append("<end_of_turn>\n<start_of_turn>model\n");
                }
            } else if (mode == 5) {
                if (!sysPromptSent && sysPrompt.length() > 0) {
                    pb.append(sysPrompt).append("\n\n");
                    sysPromptSent = true;
                }
                pb.append("### Instruction:\n").append(text).append("\n\n### Response:\n");
            } else if (mode == 6) {
                if (!sysPromptSent && sysPrompt.length() > 0) {
                    pb.append("<|system|>\n").append(sysPrompt).append("<|end|>\n");
                    sysPromptSent = true;
                }
                pb.append("<|user|>\n").append(text).append("<|end|>\n<|assistant|>\n");
            } else if (mode == 7) {
                if (!sysPromptSent && sysPrompt.length() > 0) {
                    pb.append("[INST] ").append(sysPrompt).append("\n\n").append(text).append(" [/INST]");
                    sysPromptSent = true;
                } else {
                    pb.append("[INST] ").append(text).append(" [/INST]");
                }
            } else {
                if (!sysPromptSent && sysPrompt.length() > 0) {
                    pb.append(sysPrompt).append("\n");
                    sysPromptSent = true;
                }
                pb.append("<|user|>").append(text).append("<|assistant|>");
            }
            formattedPrompt = pb.toString();
            addMessageItem("You", text, new ChatEntry());
            tvOut = addMessageItem("Model", "", outEntry);
        } else {
            formattedPrompt = text;
            addMessageItem("Prompt", text, new ChatEntry());
            tvOut = addMessageItem("Completion", text, outEntry);
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
                                    String full = outBuf.toString();
                                    outEntry.text = full;
                                    tvOut.setText(full);
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

    private void showExportDialog() {
        if (chatHistory.isEmpty()) {
            Toast.makeText(this, "Nothing to export", Toast.LENGTH_SHORT).show();
            return;
        }
        String[] options = {
            "Save TXT to Download",
            "Save JSON to Download",
            "Choose Folder & Save TXT",
            "Choose Folder & Save JSON"
        };
        new AlertDialog.Builder(this)
            .setTitle("Export Chat")
            .setItems(options, new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    boolean asJson = (which == 1 || which == 3);
                    if (which == 0 || which == 1) {
                        File sd = Environment.getExternalStorageDirectory();
                        if (sd == null || !sd.exists()) sd = getFilesDir();
                        File dl = new File(sd, "Download");
                        if (!dl.exists()) dl.mkdirs();
                        exportChatToDir(dl, asJson);
                    } else {
                        pendingExportJson = asJson;
                        Intent i = new Intent(ChatActivity.this, FilePickerActivity.class);
                        i.putExtra(FilePickerActivity.EXTRA_PICK_DIR, true);
                        startActivityForResult(i, REQ_EXPORT_DIR);
                    }
                }
            })
            .show();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQ_EXPORT_DIR && resultCode == RESULT_OK && data != null) {
            String dirPath = data.getStringExtra(FilePickerActivity.EXTRA_SELECTED_PATH);
            if (dirPath != null && dirPath.length() > 0) {
                exportChatToDir(new File(dirPath), pendingExportJson);
            }
        }
    }

    private void exportChatToDir(File dir, boolean asJson) {
        if (dir == null) return;
        if (!dir.exists()) dir.mkdirs();
        String tsName = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        File outFile = new File(dir, "indext_chat_" + tsName + (asJson ? ".json" : ".txt"));
        try {
            String content;
            if (asJson) {
                JSONObject root = new JSONObject();
                root.put("app", "InDEXt Loader 26.0.1");
                root.put("model", AppState.loadedModelName);
                root.put("exported_at", new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date()));
                JSONArray arr = new JSONArray();
                for (ChatEntry e : chatHistory) {
                    JSONObject obj = new JSONObject();
                    obj.put("role", e.role);
                    obj.put("timestamp", e.timestamp);
                    obj.put("text", e.text);
                    arr.put(obj);
                }
                root.put("messages", arr);
                content = root.toString(2);
            } else {
                StringBuilder sb = new StringBuilder();
                sb.append("Model: ").append(AppState.loadedModelName).append("\n\n");
                for (ChatEntry e : chatHistory) {
                    sb.append("[").append(e.timestamp).append("] ").append(e.role).append(":\n");
                    sb.append(e.text).append("\n\n");
                }
                content = sb.toString();
            }
            OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(outFile), "UTF-8");
            osw.write(content);
            osw.close();
            Toast.makeText(this, "Exported: " + outFile.getAbsolutePath(), Toast.LENGTH_LONG).show();
        } catch (Exception e) {
            Toast.makeText(this, "Export failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void clearContext() {
        if (isGenerating) NativeEngine.nativeRequestStop(AppState.engineHandle);
        NativeEngine.nativeResetContext(AppState.engineHandle);
        msgList.removeAllViews();
        chatHistory.clear();
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
