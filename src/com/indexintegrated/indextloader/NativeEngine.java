package com.indexintegrated.indextloader;

public final class NativeEngine {
    static {
        System.loadLibrary("indext");
    }

    public interface TokenCallback {
        int onToken(byte[] pieceBytes);
    }

    public static native String nativeGetLastError();

    public static native long nativeLoadModel(
        String modelPath,
        String tokenizerPath,
        int nThreads,
        int nCtx,
        int maxTokens,
        float temperature,
        float topP,
        int topK,
        float repeatPenalty,
        int seed
    );

    public static native void nativeFreeModel(long handle);
    public static native void nativeResetContext(long handle);
    public static native void nativeRequestStop(long handle);

    public static native void nativeUpdateParams(
        long handle,
        int nThreads,
        int maxTokens,
        float temperature,
        float topP,
        int topK,
        float repeatPenalty,
        int seed
    );

    public static native String nativeGetModelInfo(long handle);
    public static native String nativeGetLastStats(long handle);

    public static native int nativeGenerate(
        long handle,
        String prompt,
        boolean addBos,
        boolean continueCtx,
        TokenCallback callback
    );
}
