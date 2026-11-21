LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE    := indext
LOCAL_SRC_FILES := InDEX-Transformer.c gguf_parser.c tokenizer.c indext_jni.c
LOCAL_CFLAGS    := -std=c99 -O3 -ffast-math -fomit-frame-pointer -funroll-loops -DNDEBUG
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
LOCAL_ARM_MODE  := arm
LOCAL_ARM_NEON  := true
LOCAL_CFLAGS    += -mfpu=neon -mfloat-abi=softfp
endif
LOCAL_LDLIBS    := -llog -lm
include $(BUILD_SHARED_LIBRARY)
