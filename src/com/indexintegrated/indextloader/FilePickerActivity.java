package com.indexintegrated.indextloader;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

public class FilePickerActivity extends Activity {
    public static final String EXTRA_EXT_FILTER = "ext_filter";
    public static final String EXTRA_SELECTED_PATH = "selected_path";

    private TextView pathView;
    private ListView listView;
    private File currentDir;
    private String[] extFilter;
    private final List<File> entries = new ArrayList<File>();

    private int colorPrimary;
    private int colorSecondary;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        AppState.applyTheme(this);
        super.onCreate(savedInstanceState);

        colorPrimary = AppState.getThemeColor(this, android.R.attr.textColorPrimary, 0xFFFFFFFF);
        colorSecondary = AppState.getThemeColor(this, android.R.attr.textColorSecondary, 0xFFAAAAAA);

        String filterStr = getIntent().getStringExtra(EXTRA_EXT_FILTER);
        if (filterStr != null && filterStr.length() > 0) {
            extFilter = filterStr.toLowerCase(Locale.US).split(",");
        }

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(8), dp(8), dp(8), dp(8));

        LinearLayout nav = new LinearLayout(this);
        nav.setOrientation(LinearLayout.HORIZONTAL);
        nav.setPadding(0, 0, 0, dp(4));

        Button btnSd = makeNavBtn("SDCard");
        Button btnDl = makeNavBtn("Download");
        Button btnRoot = makeNavBtn("/");
        Button btnUp = makeNavBtn("Up");
        nav.addView(btnSd);
        nav.addView(btnDl);
        nav.addView(btnRoot);
        nav.addView(btnUp);
        root.addView(nav);

        pathView = new TextView(this);
        pathView.setTextSize(12);
        pathView.setTextColor(colorSecondary);
        pathView.setPadding(dp(4), dp(2), dp(4), dp(6));
        root.addView(pathView);

        listView = new ListView(this);
        root.addView(listView, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1.0f));
        setContentView(root);

        File sd = Environment.getExternalStorageDirectory();
        if (sd == null || !sd.exists()) sd = new File("/");
        File dl = new File(sd, "Download");
        int dirMode = AppState.prefs(this).getInt(AppState.KEY_DEFAULT_DIR, 0);
        if (dirMode == 2) {
            String last = AppState.prefs(this).getString(AppState.KEY_LAST_DIR, "");
            File lf = new File(last);
            currentDir = (last.length() > 0 && lf.exists() && lf.isDirectory()) ? lf : ((dl.exists() && dl.isDirectory()) ? dl : sd);
        } else if (dirMode == 1) {
            currentDir = sd;
        } else {
            currentDir = (dl.exists() && dl.isDirectory()) ? dl : sd;
        }

        final File sdFinal = sd;
        btnSd.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { loadDir(sdFinal); }
        });
        btnDl.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                File d = new File(sdFinal, "Download");
                if (!d.exists()) d = new File(sdFinal, "Downloads");
                if (d.exists() && d.isDirectory()) loadDir(d);
                else Toast.makeText(FilePickerActivity.this, "Download folder not found", Toast.LENGTH_SHORT).show();
            }
        });
        btnRoot.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { loadDir(new File("/")); }
        });
        btnUp.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                File p = currentDir.getParentFile();
                if (p != null) loadDir(p);
            }
        });

        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                if (position < 0 || position >= entries.size()) return;
                File f = entries.get(position);
                if (f.isDirectory()) {
                    loadDir(f);
                } else {
                    Intent out = new Intent();
                    out.putExtra(EXTRA_SELECTED_PATH, f.getAbsolutePath());
                    setResult(RESULT_OK, out);
                    finish();
                }
            }
        });

        loadDir(currentDir);
    }

    private Button makeNavBtn(String text) {
        Button b = new Button(this);
        b.setText(text);
        b.setTextSize(13);
        b.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));
        return b;
    }

    private boolean matchesFilter(String name) {
        if (extFilter == null || extFilter.length == 0) return true;
        String low = name.toLowerCase(Locale.US);
        for (String ext : extFilter) {
            if (low.endsWith(ext.trim())) return true;
        }
        return false;
    }

    private void loadDir(File dir) {
        if (dir == null || !dir.exists() || !dir.isDirectory()) return;
        File[] raw = dir.listFiles();
        if (raw == null) {
            Toast.makeText(this, "Access denied", Toast.LENGTH_SHORT).show();
            return;
        }
        currentDir = dir;
        AppState.prefs(this).edit().putString(AppState.KEY_LAST_DIR, dir.getAbsolutePath()).commit();
        pathView.setText(dir.getAbsolutePath());

        Arrays.sort(raw, new Comparator<File>() {
            @Override
            public int compare(File a, File b) {
                if (a.isDirectory() != b.isDirectory()) return a.isDirectory() ? -1 : 1;
                boolean ma = !a.isDirectory() && matchesFilter(a.getName());
                boolean mb = !b.isDirectory() && matchesFilter(b.getName());
                if (ma != mb) return ma ? -1 : 1;
                return a.getName().compareToIgnoreCase(b.getName());
            }
        });

        entries.clear();
        for (File f : raw) {
            if (f.getName().startsWith(".")) continue;
            if (f.isDirectory() || matchesFilter(f.getName()) || extFilter == null) {
                entries.add(f);
            }
        }
        if (entries.isEmpty() && raw.length > 0) {
            for (File f : raw) {
                if (!f.getName().startsWith(".")) entries.add(f);
            }
        }

        ArrayAdapter<File> adapter = new ArrayAdapter<File>(this, android.R.layout.simple_list_item_1, entries) {
            @Override
            public View getView(int position, View convertView, ViewGroup parent) {
                LinearLayout row;
                TextView title, sub;
                if (convertView instanceof LinearLayout) {
                    row = (LinearLayout) convertView;
                    title = (TextView) row.getChildAt(0);
                    sub = (TextView) row.getChildAt(1);
                } else {
                    row = new LinearLayout(FilePickerActivity.this);
                    row.setOrientation(LinearLayout.VERTICAL);
                    row.setPadding(dp(8), dp(9), dp(8), dp(9));
                    title = new TextView(FilePickerActivity.this);
                    title.setTextSize(15);
                    sub = new TextView(FilePickerActivity.this);
                    sub.setTextSize(12);
                    row.addView(title);
                    row.addView(sub);
                }
                File f = entries.get(position);
                if (f.isDirectory()) {
                    title.setText(f.getName() + "/");
                    title.setTypeface(null, Typeface.BOLD);
                    title.setTextColor(colorPrimary);
                    sub.setText("Directory");
                    sub.setTextColor(colorSecondary);
                } else {
                    boolean match = matchesFilter(f.getName());
                    title.setText(f.getName());
                    title.setTypeface(null, match ? Typeface.BOLD : Typeface.NORMAL);
                    title.setTextColor(colorPrimary);
                    sub.setText(formatSize(f.length()));
                    sub.setTextColor(colorSecondary);
                }
                return row;
            }
        };
        listView.setAdapter(adapter);
    }

    private String formatSize(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format(Locale.US, "%.1f KB", bytes / 1024.0);
        return String.format(Locale.US, "%.2f MB", bytes / (1024.0 * 1024.0));
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density + 0.5f);
    }
}
