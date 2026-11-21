package com.indexintegrated.indextloader;

import android.app.Activity;
import android.app.ActivityManager;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.res.TypedArray;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.text.InputType;
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.PopupMenu;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.util.Locale;

public final class UiHelper {
    public static final int MENU_OVERFLOW_ID = 9000;

    public interface OverflowAction {
        void onSelect(int itemId);
    }

    public static void setActionBarSubtitle(Activity act, String subtitle) {
        if (Build.VERSION.SDK_INT >= 11 && act.getActionBar() != null) {
            act.getActionBar().setSubtitle(subtitle);
        }
    }

    public static void addOverflowButton(Menu menu) {
        MenuItem item = menu.add(0, MENU_OVERFLOW_ID, 999, "Menu");
        item.setIcon(android.R.drawable.ic_menu_more);
        if (Build.VERSION.SDK_INT >= 11) {
            item.setShowAsAction(MenuItem.SHOW_AS_ACTION_ALWAYS);
        }
    }

    public static void showOverflowPopup(final Activity act, final int[] ids, final String[] labels, final OverflowAction action) {
        if (Build.VERSION.SDK_INT >= 11) {
            View anchor = act.findViewById(MENU_OVERFLOW_ID);
            if (anchor == null) {
                anchor = act.getWindow().getDecorView().findViewById(android.R.id.content);
            }
            if (anchor != null) {
                PopupMenu pm = new PopupMenu(act, anchor);
                Menu m = pm.getMenu();
                for (int i = 0; i < ids.length; i++) {
                    m.add(0, ids[i], i, labels[i]);
                }
                pm.setOnMenuItemClickListener(new PopupMenu.OnMenuItemClickListener() {
                    @Override
                    public boolean onMenuItemClick(MenuItem item) {
                        action.onSelect(item.getItemId());
                        return true;
                    }
                });
                pm.show();
                return;
            }
        }
        new AlertDialog.Builder(act)
            .setItems(labels, new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    if (which >= 0 && which < ids.length) action.onSelect(ids[which]);
                }
            })
            .show();
    }

    public static void checkMemoryAndWarn(Context ctx, String modelPath) {
        if (!AppState.prefs(ctx).getBoolean(AppState.KEY_MEM_WARN, true)) return;
        if (modelPath == null || modelPath.length() == 0) return;
        File f = new File(modelPath);
        if (!f.exists()) return;
        long fileBytes = f.length();

        ActivityManager am = (ActivityManager) ctx.getSystemService(Context.ACTIVITY_SERVICE);
        if (am == null) return;
        ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
        am.getMemoryInfo(mi);

        long avail = mi.availMem;
        if (mi.lowMemory || fileBytes > (avail * 3L) / 4L || fileBytes > 900L * 1024L * 1024L) {
            Toast.makeText(
                ctx,
                String.format(Locale.US, "Warning: Low memory (%s free) for %s. Model may fail or lag.",
                    formatFileSize(avail), formatFileSize(fileBytes)),
                Toast.LENGTH_LONG
            ).show();
        }
    }

    public static Drawable getSystemCopyIcon(Activity act) {
        if (Build.VERSION.SDK_INT >= 11) {
            TypedArray ta = act.obtainStyledAttributes(new int[] { android.R.attr.actionModeCopyDrawable });
            Drawable d = ta.getDrawable(0);
            ta.recycle();
            return d;
        }
        return null;
    }

    public static String formatFileSize(long bytes) {
        if (bytes <= 0) return "0 B";
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format(Locale.US, "%.1f KB", bytes / 1024.0);
        return String.format(Locale.US, "%.1f MB", bytes / (1024.0 * 1024.0));
    }

    public static void showModelInfoDialog(Activity act) {
        if (AppState.engineHandle == 0) {
            new AlertDialog.Builder(act)
                .setTitle("Model Info")
                .setMessage("No model is currently loaded in memory.")
                .setPositiveButton("OK", null)
                .show();
            return;
        }
        String[] inf = AppState.loadedModelInfo.split("\\|");
        StringBuilder sb = new StringBuilder();
        sb.append("File: ").append(AppState.loadedModelName).append("\n");
        if (inf.length >= 10) {
            sb.append("Format: ").append(inf[0]).append("\n");
            sb.append("Architecture: ").append(inf[1]).append("\n");
            sb.append("Vocab: ").append(inf[2]).append("\n");
            sb.append("Dim: ").append(inf[3]).append(" | Layers: ").append(inf[4]).append("\n");
            sb.append("Heads: ").append(inf[5]).append(" (KV: ").append(inf[6]).append(")\n");
            sb.append("Context: ").append(inf[7]).append("\n");
            sb.append("File Size: ").append(inf[8]).append(" MB\n");
            String kvMode = "FP32";
            if (inf.length >= 13) {
                if ("1".equals(inf[12])) kvMode = "FP16";
                else if ("2".equals(inf[12])) kvMode = "Q8_0";
                else if ("3".equals(inf[12])) kvMode = "Q4_0";
            }
            sb.append("KV Cache: ").append(inf[9]).append(" MB (").append(kvMode).append(")\n");
            if (inf.length >= 12) {
                sb.append("Storage Mode: ").append("1".equals(inf[11]) ? "mmap (zero-copy)" : "RAM buffer");
            }
        }
        new AlertDialog.Builder(act)
            .setTitle("Model Info")
            .setMessage(sb.toString())
            .setPositiveButton("OK", null)
            .show();
    }

    public static int dp(Activity act, int v) {
        return (int) (v * act.getResources().getDisplayMetrics().density + 0.5f);
    }

    public static CheckBox makeCheck(Activity act, LinearLayout parent, String label, boolean checked) {
        CheckBox cb = new CheckBox(act);
        cb.setText(label);
        cb.setChecked(checked);
        parent.addView(cb);
        return cb;
    }

    public static TextView sectionHeader(Activity act, String text, int color) {
        TextView tv = new TextView(act);
        tv.setText(text);
        tv.setTextSize(12);
        tv.setTypeface(null, Typeface.BOLD);
        tv.setTextColor(color);
        tv.setPadding(0, dp(act, 10), 0, dp(act, 4));
        return tv;
    }

    public static Spinner makeSpinnerRow(Activity act, LinearLayout parent, String label, String[] items, int selIdx, int labelColor) {
        LinearLayout row = new LinearLayout(act);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(act, 2), 0, dp(act, 2));

        TextView tv = new TextView(act);
        tv.setText(label);
        tv.setTextSize(14);
        tv.setTextColor(labelColor);
        row.addView(tv, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));

        Spinner sp = new Spinner(act);
        ArrayAdapter<String> ad = new ArrayAdapter<String>(act, android.R.layout.simple_spinner_item, items);
        ad.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        sp.setAdapter(ad);
        if (selIdx >= 0 && selIdx < items.length) sp.setSelection(selIdx);
        row.addView(sp, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));
        parent.addView(row);
        return sp;
    }

    public static EditText makeEditRow(Activity act, LinearLayout parent, String label, String defVal, int labelColor) {
        LinearLayout row = new LinearLayout(act);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(act, 2), 0, dp(act, 2));

        TextView tv = new TextView(act);
        tv.setText(label);
        tv.setTextSize(14);
        tv.setTextColor(labelColor);
        row.addView(tv, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.2f));

        EditText et = new EditText(act);
        et.setText(defVal);
        et.setSingleLine(true);
        et.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL);
        row.addView(et, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 0.8f));
        parent.addView(row);
        return et;
    }
}
