import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
NDK_BUILD = r"C:\Android\ndk\android-ndk-r16b\ndk-build.cmd"
ANDROID_JAR = r"C:\Android\sdk\platforms\android-19\android.jar"
BT = r"C:\Android\sdk\build-tools\28.0.3"
AAPT = os.path.join(BT, "aapt.exe")
DX = os.path.join(BT, "dx.bat")
ZIPALIGN = os.path.join(BT, "zipalign.exe")
APKSIGNER = os.path.join(BT, "apksigner.bat")

GEN_DIR = os.path.join(ROOT, "gen")
OBJ_DIR = os.path.join(ROOT, "obj_java")
BUILD_DIR = os.path.join(ROOT, "build")
OUT_APK = os.path.join(ROOT, "InDEXtLoader-2026.0.0.apk")
KEYSTORE = os.path.join(ROOT, "debug.keystore")

def run(cmd, cwd=ROOT):
    print(">>", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        sys.exit(res.returncode)

def main():
    for d in [GEN_DIR, OBJ_DIR, BUILD_DIR]:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    # 1. Native libraries via NDK r16b
    run([NDK_BUILD, "-j4"])

    # 2. Generate R.java
    run([AAPT, "package", "-f", "-m", "-J", GEN_DIR, "-M", "AndroidManifest.xml", "-S", "res", "-I", ANDROID_JAR])

    # 3. Compile Java (pure Java 7 compatible syntax, compiled with -source 8 -target 8)
    java_files = []
    for base in [GEN_DIR, os.path.join(ROOT, "src")]:
        for dp, _, fns in os.walk(base):
            for fn in fns:
                if fn.endswith(".java"):
                    java_files.append(os.path.join(dp, fn))
    run(["javac", "-encoding", "UTF-8", "-source", "8", "-target", "8", "-bootclasspath", ANDROID_JAR, "-d", OBJ_DIR] + java_files)

    # 4. Convert to classes.dex (call dx.jar directly for JDK 17 compatibility)
    dex_path = os.path.join(BUILD_DIR, "classes.dex")
    dx_jar = os.path.join(BT, "lib", "dx.jar")
    run(["java", "-Xmx1024M", "-jar", dx_jar, "--dex", "--output=" + dex_path, OBJ_DIR])

    # 5. Package resources
    unaligned_apk = os.path.join(BUILD_DIR, "unaligned.apk")
    run([AAPT, "package", "-f", "-M", "AndroidManifest.xml", "-S", "res", "-I", ANDROID_JAR, "-F", unaligned_apk])

    # 6. Inject classes.dex and native .so libs into APK
    with zipfile.ZipFile(unaligned_apk, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.write(dex_path, "classes.dex")
        for abi in ["armeabi", "armeabi-v7a"]:
            so_src = os.path.join(ROOT, "libs", abi, "libindext.so")
            if os.path.exists(so_src):
                zf.write(so_src, f"lib/{abi}/libindext.so")

    # 7. Zipalign
    aligned_apk = os.path.join(BUILD_DIR, "aligned.apk")
    run([ZIPALIGN, "-f", "4", unaligned_apk, aligned_apk])

    # 8. Debug keystore + v1 sign (required for Android 2.3 - 4.4)
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
