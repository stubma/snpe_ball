#include <getopt.h>
#include <stdio.h>
#include "log.h"
#include "codec_api.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include "dlc_runner.h"
#include "global.h"

static const int ARG_VERSION = 'v';
static const int ARG_DSP_LIB_DIR = 'l';
static const int ARG_HELP = 'h';
static const int ARG_DUMP_FORMAT = 'd';
static const int ARG_VIDEO_WIDTH = 1000;
static const int ARG_VIDEO_HEIGHT = 1001;
static const char short_options[] = "d:hl:v";
static const struct option long_options[] = {
        {"version", no_argument, NULL, ARG_VERSION},
        {"dsp_library_path", required_argument, NULL, ARG_DSP_LIB_DIR},
        {"help", no_argument, NULL, ARG_HELP},
        {"dump", required_argument, NULL, ARG_DUMP_FORMAT},
        {"vw", required_argument, NULL, ARG_VIDEO_WIDTH},
        {"video_width", required_argument, NULL, ARG_VIDEO_WIDTH},
        {"vh", required_argument, NULL, ARG_VIDEO_HEIGHT},
        {"video_height", required_argument, NULL, ARG_VIDEO_HEIGHT},
        {0, 0, 0, 0}
};

static void print_version() {
    DlSystem::Version_t libVer = SNPE::SNPEFactory::getLibraryVersion();
    printf("SNPE version: %s, available runtime: %s, opencv version: %s\n",
           libVer.toString().c_str(),
           getRuntimeStr().c_str(),
           CV_VERSION);
}

static void print_usage(int argc, char *argv[]) {
    fprintf(stdout,
            "Usage: %s [options]\n"
            "Version %s\n"
            "Options:\n"
            "-d | --dump [yuv|jpg]\t\tdump frame in specified format, can be yuv or jpg\n"
            "-h | --help\t\t\tprint help\n"
            "-l | --dsp_library_path\t\tset ADSP_LIBRARY_PATH environment, default is /vendor/lib\n"
            "-v | --version\t\t\tprint version \n"
            "--vw | --video_width\t\tspecify video width\n"
            "--vh | --video_height\t\tspecify video height\n"
            "\n",
            argv[0], "v1.0.0");
}

static void process_opt(int argc, char *argv[]) {
    for (;;) {
        int idx;
        int c = getopt_long(argc, argv, short_options, long_options, &idx);
        if (-1 == c)
            break;
        switch (c) {
            case ARG_DUMP_FORMAT:
                if(!strcmp(optarg, "yuv")) {
                    g_output_file_type = REWOO_OUTPUT_YUV;
                } else if(!strcmp(optarg, "jpg")) {
                    g_output_file_type = REWOO_OUTPUT_JPG;
                }
                break;
            case ARG_HELP:
                print_usage(argc, argv);
                exit(EXIT_SUCCESS);
            case ARG_VERSION:
                print_version();
                exit(EXIT_SUCCESS);
            case ARG_DSP_LIB_DIR:
                g_dsp_lib_dir = optarg;
                setenv(DSP_ENV_VAR, optarg, true);
                break;
            case ARG_VIDEO_WIDTH:
                g_video_width = atoi(optarg);
                break;
            case ARG_VIDEO_HEIGHT:
                g_video_height = atoi(optarg);
                break;
            default:
                break;
        }
    }
}

static void onOutputAvailable(
        AMediaCodec *codec,
        Decoder *decoder,
        int32_t index,
        AMediaCodecBufferInfo *bufferInfo) {
    size_t bufSize;
    uint8_t *buf = AMediaCodec_getOutputBuffer(codec, index, &bufSize);
    if (buf && bufferInfo->size > 0) {
        char path[512] = {0};
        switch(g_output_file_type) {
            case REWOO_OUTPUT_YUV: {
                sprintf(path, "%s/frame_%d.yuv", g_output_dir.c_str(), decoder->getOuputFrameNum());
                FILE *fp = fopen(path, "w+");
                fwrite(buf, sizeof(char), bufferInfo->size, fp);
                fflush(fp);
                fclose(fp);
                ALOGV("bytes(%d) written into file %s", bufferInfo->size, path);
                break;
            }
            case REWOO_OUTPUT_JPG: {
                sprintf(path, "%s/frame_%d.jpg", g_output_dir.c_str(), decoder->getOuputFrameNum());
                cv::Mat matSrc = cv::Mat(g_video_height * 1.5, g_video_width, CV_8UC1, buf);
                cv::Mat matDst = cv::Mat(g_video_height, g_video_width, CV_8UC3);
                cv::cvtColor(matSrc, matDst, cv::COLOR_YUV2RGB_NV21);
                cv::imwrite(path, matDst);
                ALOGV("JPG written into file %s", path);
                break;
            }
            default:
                break;
        }
    }
}

int main(int argc, char *argv[]) {
    // set dsp library path so that runtime can use dsp
    setenv(DSP_ENV_VAR, DEFAULT_DSP_LIB_DIR, true);

    // handle arguments
    process_opt(argc, argv);

    // ensure output dir exist
    if(!g_output_dir.empty() && !is_directory(g_output_dir)) {
        mkdirs(g_output_dir.c_str());
    }

    RewooDecoderCallback cb{
            nullptr,
            onOutputAvailable,
            nullptr,
            nullptr
    };
    decode_video(
            "/data/local/tmp/MediaBenchmark/res/",
            "rewoo_full.mp4",
            "/data/local/tmp/decoder.stat",
            "c2.qti.avc.decoder",
            false,
            &cb,
            nullptr);

    // ok
    return EXIT_SUCCESS;
}