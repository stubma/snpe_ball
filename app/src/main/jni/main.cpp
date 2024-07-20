#include <getopt.h>
#include <stdio.h>
#include "log.h"
#include "codec_api.h"
#include "shared/utils.h"
#include <opencv2/opencv.hpp>
#include "shared/global.h"
#include "tensor_producer.h"
#include "tensor_consumer.h"
#include <linux/limits.h>
#include "param.h"

static const int ARG_VERSION = 'v';
static const int ARG_DSP_LIB_DIR = 'l';
static const int ARG_HELP = 'h';
static const int ARG_DUMP_FORMAT = 'd';
static const int ARG_VIDEO_WIDTH = 1000;
static const int ARG_VIDEO_HEIGHT = 1001;
static const int ARG_DLC_PATH = 1004;
static const int ARG_VIDEO_PATH = 1006;
static const int ARG_OUTPUT_DIR = 1007;
static const char short_options[] = "d:hl:v";
static const struct option long_options[] = {
        {"version", no_argument, NULL, ARG_VERSION},
        {"dsp_library_path", required_argument, NULL, ARG_DSP_LIB_DIR},
        {"help", no_argument, NULL, ARG_HELP},
        {"dump", required_argument, NULL, ARG_DUMP_FORMAT},
        {"vp", required_argument, NULL, ARG_VIDEO_PATH},
        {"video_path", required_argument, NULL, ARG_VIDEO_PATH},
        {"vw", required_argument, NULL, ARG_VIDEO_WIDTH},
        {"video_width", required_argument, NULL, ARG_VIDEO_WIDTH},
        {"vh", required_argument, NULL, ARG_VIDEO_HEIGHT},
        {"video_height", required_argument, NULL, ARG_VIDEO_HEIGHT},
        {"dlc", required_argument, NULL, ARG_DLC_PATH},
        {"output_path", required_argument, NULL, ARG_OUTPUT_DIR},
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
                if (!strcmp(optarg, "yuv")) {
                    g_output_file_type = REWOO_OUTPUT_YUV;
                } else if (!strcmp(optarg, "jpg")) {
                    g_output_file_type = REWOO_OUTPUT_JPG;
                } else if (!strcmp(optarg, "raw")) {
                    g_output_file_type = REWOO_OUTPUT_RAW_RGB;
                } else {
                    g_output_file_type = REWOO_OUTPUT_NONE;
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
                if (!starts_with(g_dsp_lib_dir, "/")) {
                    g_dsp_lib_dir = g_cwd + "/" + g_dsp_lib_dir;
                }
                setenv(DSP_ENV_VAR, g_dsp_lib_dir.c_str(), true);
                break;
            case ARG_VIDEO_PATH:
                g_video_path = optarg;
                if (!starts_with(g_video_path, "/")) {
                    g_video_path = g_cwd + "/" + g_video_path;
                }
                break;
            case ARG_VIDEO_WIDTH:
                g_video_width = atoi(optarg);
                break;
            case ARG_VIDEO_HEIGHT:
                g_video_height = atoi(optarg);
                break;
            case ARG_DLC_PATH:
                g_dlc_path = optarg;
                if (!starts_with(g_dlc_path, "/")) {
                    g_dlc_path = g_cwd + "/" + g_dlc_path;
                }
                g_dlc_dir = remove_last_path_component(g_dlc_path);
                break;
            case ARG_OUTPUT_DIR:
                g_output_dir = optarg;
                if (!starts_with(g_output_dir, "/")) {
                    g_output_dir = g_cwd + "/" + g_output_dir;
                }
                break;
            default:
                break;
        }
    }
}

static void loadConfig() {
    // init config
    std::string ini_path = g_cwd + "/cfg.ini";
    if(!is_file_exists(ini_path)) {
        ALOGD("please place cfg.ini in the same directory of executable");
        exit(EXIT_FAILURE);
        return;
    }
    rk_param_init((char*)ini_path.c_str());

    // load
    g_dsp_lib_dir = rk_param_get_string("snpe:dsp_lib_dir", "lib");
    if (!starts_with(g_dsp_lib_dir, "/")) {
        g_dsp_lib_dir = g_cwd + "/" + g_dsp_lib_dir;
    }
    g_dlc_path = rk_param_get_string("snpe:model_path", "");
    if (!starts_with(g_dlc_path, "/")) {
        g_dlc_path = g_cwd + "/" + g_dlc_path;
    }
    g_dlc_dir = remove_last_path_component(g_dlc_path);
    g_video_path = rk_param_get_string("input:video_path", "");
    if (!starts_with(g_video_path, "/")) {
        g_video_path = g_cwd + "/" + g_video_path;
    }
    g_video_codec = rk_param_get_string("input:video_codec", "c2.qti.avc.decoder");
    g_video_width = rk_param_get_int("input:video_width", 7600);
    g_video_height = rk_param_get_int("input:video_height", 2160);
    float points[8] = {0};
    int num;
    rk_param_get_float_array("input:left_goalnet_points", points, 8, &num);
    g_goalnet_points.clear();
    g_goalnet_points.push_back({(int)points[0], (int)points[1]});
    g_goalnet_points.push_back({(int)points[2], (int)points[3]});
    g_goalnet_points.push_back({(int)points[4], (int)points[5]});
    g_goalnet_points.push_back({(int)points[6], (int)points[7]});
    bool enable_output = rk_param_get_int("output:enable", 0);
    if(enable_output) {
        const char* type = rk_param_get_string("output:format", "jpg");
        if(!strcmp(type, "yuv")) {
            g_output_file_type = REWOO_OUTPUT_YUV;
        } else if(!strcmp(type, "jpg")) {
            g_output_file_type = REWOO_OUTPUT_JPG;
        } else {
            g_output_file_type = REWOO_OUTPUT_RAW_RGB;
        }
    } else {
        g_output_file_type = REWOO_OUTPUT_NONE;
    }
    g_output_dir = rk_param_get_string("output:folder", "decode_output");
    if (!starts_with(g_output_dir, "/")) {
        g_output_dir = g_cwd + "/" + g_output_dir;
    }
    g_output_from_frame = rk_param_get_int("output:from", 1);
    g_output_to_frame = rk_param_get_int("output:to", -1);
}

static void dumpConfig() {
    ALOGD("======== dump configuration start ========>");
    ALOGD("dsp library path: %s", g_dsp_lib_dir.c_str());
    ALOGD("model path: %s", g_dlc_path.c_str());
    ALOGD("input video path: %s, resolution: %dx%d", g_video_path.c_str(), g_video_width, g_video_height);
    ALOGD("input video codec: %s", g_video_codec.c_str());
    ALOGD("video left goalnet points: (%d, %d) - (%d, %d) - (%d, %d) - (%d, %d)",
          g_goalnet_points[0].x, g_goalnet_points[0].y,
          g_goalnet_points[1].x, g_goalnet_points[1].y,
          g_goalnet_points[2].x, g_goalnet_points[2].y,
          g_goalnet_points[3].x, g_goalnet_points[3].y);
    ALOGD("video dump enabled: %s", g_output_file_type == REWOO_OUTPUT_NONE ? "false" : "true");
    if(g_output_file_type != REWOO_OUTPUT_NONE) {
        ALOGD("video dump dir: %s", g_output_dir.c_str());
        ALOGD("video dump format: %s", get_output_type_str(g_output_file_type).c_str());
        ALOGD("video dump frame range: [%d, %d]", g_output_from_frame, g_output_to_frame);
    }
    ALOGD("========= dump configuration end   <========\n\n");
}

int main(int argc, char *argv[]) {
    // get current directory
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        g_cwd = cwd;
    }

    // load config
    loadConfig();

    // set dsp library path so that runtime can use dsp
    setenv(DSP_ENV_VAR, g_dsp_lib_dir.c_str(), true);

    // handle arguments
    process_opt(argc, argv);

    // dump
    dumpConfig();

    // check video path
    if(!is_file_exists(g_video_path)) {
        ALOGD("video file %s doesn't not exist", g_video_path.c_str());
        return EXIT_FAILURE;
    }

    // ensure output dir exist
    if (!g_output_dir.empty() && !is_directory(g_output_dir)) {
        mkdirs(g_output_dir.c_str());
    }

    // 如果指定了模型路径, 则进入模型运行逻辑
    // 如果没有指定模型路径, 则进入视频解码测试逻辑
    if (!g_dlc_path.empty()) {
        // run consumer
        TensorConsumer c;

        // run producer
        TensorProducer p(&c);
        p.run();

        // run consumer
        c.run();

        // wait done
        const auto start = std::chrono::high_resolution_clock::now();
        while (!g_decode_done || !g_dlc_done) {
            sleep(1);
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::milliseconds cost = std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start);
        ALOGD("total running time: %lldms", cost.count());
    }

    // ok
    return EXIT_SUCCESS;
}