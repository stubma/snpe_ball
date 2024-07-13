#pragma once

#ifdef __ANDROID__

#include <android/log.h>

#define __logv(tag, mArgs...) __android_log_print(ANDROID_LOG_VERBOSE, tag, ##mArgs)
#define __logd(tag, mArgs...) __android_log_print(ANDROID_LOG_DEBUG, tag, ##mArgs)
#define __logi(tag, mArgs...) __android_log_print(ANDROID_LOG_INFO, tag, ##mArgs)
#define __logw(tag, mArgs...) __android_log_print(ANDROID_LOG_WARN, tag, ##mArgs)
#define __loge(tag, mArgs...) __android_log_print(ANDROID_LOG_ERROR, tag, ##mArgs)
#define __logf(tag, mArgs...) __android_log_print(ANDROID_LOG_FATAL, tag, ##mArgs)

#ifdef DEBUG
#define LOG_TAG "hexagon"
#endif

#ifdef LOG_TAG
    #ifdef STANDALONE
    #define ALOGV(...) do { printf(__VA_ARGS__); printf("\n"); } while(false)
    #define ALOGD(...) do { printf(__VA_ARGS__); printf("\n"); } while(false)
    #define ALOGI(...) do { printf(__VA_ARGS__); printf("\n"); } while(false)
    #define ALOGW(...) do { printf(__VA_ARGS__); printf("\n"); } while(false)
    #define ALOGE(...) do { printf(__VA_ARGS__); printf("\n"); } while(false)
    #define ALOGF(...) do { printf(__VA_ARGS__); printf("\n"); } while(false)
    #else
    #define ALOGV(...) __logv(LOG_TAG, __VA_ARGS__)
    #define ALOGD(...) __logd(LOG_TAG, __VA_ARGS__)
    #define ALOGI(...) __logi(LOG_TAG, __VA_ARGS__)
    #define ALOGW(...) __logw(LOG_TAG, __VA_ARGS__)
    #define ALOGE(...) __loge(LOG_TAG, __VA_ARGS__)
    #define ALOGF(...) __logf(LOG_TAG, __VA_ARGS__)
    #endif
#else
# define ALOGV(...) ((void)0)
# define ALOGD(...) ((void)0)
# define ALOGI(...) ((void)0)
# define ALOGW(...) ((void)0)
# define ALOGE(...) ((void)0)
# define ALOGF(...) ((void)0)
#endif

#endif // #ifdef __ANDROID__