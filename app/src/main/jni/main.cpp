#include <getopt.h>
#include <stdio.h>
#include "log.h"
#include "shared/utils.h"
#include <opencv2/opencv.hpp>
#include "shared/global.h"
#include "tensor_producer.h"
#include "tensor_consumer.h"
#include <linux/limits.h>
#include "param.h"

static const int ARG_VERSION = 'v';
static const char short_options[] = "v";
static const struct option long_options[] = {
        {"version", no_argument, NULL, ARG_VERSION},
        {0, 0, 0, 0}
};

static void print_version() {
    DlSystem::Version_t libVer = SNPE::SNPEFactory::getLibraryVersion();
    printf("SNPE version: %s, available runtime: %s, opencv version: %s\n",
           libVer.toString().c_str(),
           getRuntimeStr().c_str(),
           CV_VERSION);
}

static void process_opt(int argc, char *argv[]) {
    for (;;) {
        int idx;
        int c = getopt_long(argc, argv, short_options, long_options, &idx);
        if (-1 == c)
            break;
        switch (c) {
            case ARG_VERSION:
                print_version();
                exit(EXIT_SUCCESS);
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
    bool enable_dump = rk_param_get_int("dump:enable", 0);
    if(enable_dump) {
        const char* type = rk_param_get_string("dump:format", "jpg");
        if(!strcmp(type, "yuv")) {
            g_dump_file_type = REWOO_DUMP_YUV;
        } else if(!strcmp(type, "jpg")) {
            g_dump_file_type = REWOO_DUMP_JPG;
        } else {
            g_dump_file_type = REWOO_DUMP_RAW_RGB;
        }
    } else {
        g_dump_file_type = REWOO_DUMP_NONE;
    }
    g_dump_dir = rk_param_get_string("dump:folder", "decode_output");
    if (!starts_with(g_dump_dir, "/")) {
        g_dump_dir = g_cwd + "/" + g_dump_dir;
    }
    g_dump_from_frame = rk_param_get_int("dump:from", 1);
    g_dump_to_frame = rk_param_get_int("dump:to", -1);
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
    ALOGD("video dump enabled: %s", g_dump_file_type == REWOO_DUMP_NONE ? "false" : "true");
    if(g_dump_file_type != REWOO_DUMP_NONE) {
        ALOGD("video dump dir: %s", g_dump_dir.c_str());
        ALOGD("video dump format: %s", get_dump_type_str(g_dump_file_type).c_str());
        ALOGD("video dump frame range: [%d, %d]", g_dump_from_frame, g_dump_to_frame);
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

    // check dlc path
    if(!is_file_exists(g_dlc_path)) {
        ALOGD("model file %s doesn't not exist", g_dlc_path.c_str());
        return EXIT_FAILURE;
    }

    // ensure output dir exist
    if (!g_dump_dir.empty() && !is_directory(g_dump_dir)) {
        mkdirs(g_dump_dir.c_str());
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