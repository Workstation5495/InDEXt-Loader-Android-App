import os
import shutil
import struct
import subprocess
import sys
import zipfile
import zlib

ROOT = os.path.dirname(os.path.abspath(__file__))
NDK_BUILD = r"C:\Android\ndk\android-ndk-r16b\ndk-build.cmd"
ANDROID_JAR = r"C:\Android\sdk\platforms\android-19\android.jar"
BT = r"C:\Android\sdk\build-tools\28.0.3"
AAPT = os.path.join(BT, "aapt.exe")
ZIPALIGN = os.path.join(BT, "zipalign.exe")

GEN_DIR = os.path.join(ROOT, "gen")
OBJ_DIR = os.path.join(ROOT, "obj_java")
BUILD_DIR = os.path.join(ROOT, "build")
OUT_APK = os.path.join(ROOT, "InDEXtLoader-26.0.1.apk")
KEYSTORE = os.path.join(ROOT, "debug.keystore")

def run(cmd, cwd=ROOT):
    print(">>", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        sys.exit(res.returncode)

def ensure_icons():
    legacy_res = os.path.join(os.path.dirname(ROOT), "InDEXtLoader", "res")
    for d, sz in [("drawable-mdpi", 48), ("drawable-hdpi", 72), ("drawable-xhdpi", 96)]:
        dst = os.path.join(ROOT, "res", d, "ic_launcher.png")
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        src = os.path.join(legacy_res, d, "ic_launcher.png")
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            rows = []
            for y in range(sz):
                row = bytearray([0])
                m = sz // 8
                for x in range(sz):
                    row.extend([24, 118, 210, 255] if (m <= x < sz - m and m <= y < sz - m) else [0, 0, 0, 0])
                rows.append(bytes(row))
            raw = b"".join(rows)
            def chunk(tag, data):
                return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            ihdr = struct.pack(">IIBBBBB", sz, sz, 8, 6, 0, 0, 0)
            with open(dst, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

def main():
    ensure_icons()
    for d in [GEN_DIR, OBJ_DIR, BUILD_DIR]:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    run([NDK_BUILD, "-j4"])
    run([AAPT, "package", "-f", "-m", "-J", GEN_DIR, "-M", "AndroidManifest.xml", "-S", "res", "-I", ANDROID_JAR])

    java_files = []
    for base in [GEN_DIR, os.path.join(ROOT, "src")]:
        for dp, _, fns in os.walk(base):
            for fn in fns:
                if fn.endswith(".java"):
                    java_files.append(os.path.join(dp, fn))
    run(["javac", "-encoding", "UTF-8", "-source", "8", "-target", "8", "-bootclasspath", ANDROID_JAR, "-d", OBJ_DIR] + java_files)

    dex_path = os.path.join(BUILD_DIR, "classes.dex")
    dx_jar = os.path.join(BT, "lib", "dx.jar")
    run(["java", "-Xmx1024M", "-jar", dx_jar, "--dex", "--output=" + dex_path, OBJ_DIR])

    unaligned_apk = os.path.join(BUILD_DIR, "unaligned.apk")
    run([AAPT, "package", "-f", "-M", "AndroidManifest.xml", "-S", "res", "-I", ANDROID_JAR, "-F", unaligned_apk])

    with zipfile.ZipFile(unaligned_apk, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.write(dex_path, "classes.dex")
        for abi in ["armeabi", "armeabi-v7a"]:
            so_src = os.path.join(ROOT, "libs", abi, "libindext.so")
            if os.path.exists(so_src):
                zf.write(so_src, f"lib/{abi}/libindext.so")

    aligned_apk = os.path.join(BUILD_DIR, "aligned.apk")
    run([ZIPALIGN, "-f", "4", unaligned_apk, aligned_apk])

    if not os.path.exists(KEYSTORE):
        run([
            "keytool", "-genkeypair", "-v",
            "-keystore", KEYSTORE,
            "-alias", "indextdebug",
            "-keyalg", "RSA", "-keysize", "2048", "-validity", "10000",
            "-storepass", "android", "-keypass", "android",
            "-dname", "CN=InDEX Integrated,O=InDEX,C=US"
        ])

    apksigner_jar = os.path.join(BT, "lib", "apksigner.jar")
    run([
        "java", "-jar", apksigner_jar, "sign",
        "--ks", KEYSTORE,
        "--ks-pass", "pass:android",
        "--key-pass", "pass:android",
        "--ks-key-alias", "indextdebug",
        "--v1-signing-enabled", "true",
        "--v2-signing-enabled", "false",
        "--out", OUT_APK,
        aligned_apk
    ])
    print(f"\n[SUCCESS] APK ready: {OUT_APK} ({os.path.getsize(OUT_APK)} bytes)")

if __name__ == "__main__":
    main()
