#include <getopt.h>
#include <stdio.h>
#include "log.h"
#include "codec_api.h"
#include "shared/utils.h"
#include <opencv2/opencv.hpp>
#include "dlc_runner.h"
#include "shared/global.h"
#include "raw_list_provider.h"
#include "tensor_producer.h"
#include "IDlContainer.hpp"
#include "SNPE.hpp"
#include "SNPEFactory.hpp"
#include "SNPEBuilder.hpp"
#include "SetBuilderOptions.hpp"
#include "LoadContainer.hpp"

static const int ARG_VERSION = 'v';
static const int ARG_DSP_LIB_DIR = 'l';
static const int ARG_HELP = 'h';
static const int ARG_DUMP_FORMAT = 'd';
static const int ARG_VIDEO_WIDTH = 1000;
static const int ARG_VIDEO_HEIGHT = 1001;
static const int ARG_OUTPUT_WIDTH = 1002;
static const int ARG_OUTPUT_HEIGHT = 1003;
static const int ARG_DLC_PATH = 1004;
static const int ARG_INPUT_LIST = 1005;
static const int ARG_VIDEO_PATH = 1006;
static const char short_options[] = "d:hl:v";
static const struct option long_options[] = {
        {"version",          no_argument,       NULL, ARG_VERSION},
        {"dsp_library_path", required_argument, NULL, ARG_DSP_LIB_DIR},
        {"help",             no_argument,       NULL, ARG_HELP},
        {"dump",             required_argument, NULL, ARG_DUMP_FORMAT},
        {"vp",               required_argument, NULL, ARG_VIDEO_PATH},
        {"video_path",       required_argument, NULL, ARG_VIDEO_PATH},
        {"vw",               required_argument, NULL, ARG_VIDEO_WIDTH},
        {"video_width",      required_argument, NULL, ARG_VIDEO_WIDTH},
        {"vh",               required_argument, NULL, ARG_VIDEO_HEIGHT},
        {"video_height",     required_argument, NULL, ARG_VIDEO_HEIGHT},
        {"ow",               required_argument, NULL, ARG_OUTPUT_WIDTH},
        {"output_width",     required_argument, NULL, ARG_OUTPUT_WIDTH},
        {"oh",               required_argument, NULL, ARG_OUTPUT_HEIGHT},
        {"output_height",    required_argument, NULL, ARG_OUTPUT_HEIGHT},
        {"dlc",              required_argument, NULL, ARG_DLC_PATH},
        {"input_list",       required_argument, NULL, ARG_INPUT_LIST},
        {0, 0, 0,                                     0}
};
static std::string inputListFileName = "target_raw_list.txt";

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
            case ARG_VIDEO_PATH:
                g_video_path = optarg;
                break;
            case ARG_VIDEO_WIDTH:
                g_video_width = atoi(optarg);
                break;
            case ARG_VIDEO_HEIGHT:
                g_video_height = atoi(optarg);
                break;
            case ARG_OUTPUT_WIDTH:
                g_output_width = atoi(optarg);
                break;
            case ARG_OUTPUT_HEIGHT:
                g_output_height = atoi(optarg);
                break;
            case ARG_DLC_PATH:
                g_dlc_path = optarg;
                g_dlc_dir = remove_last_path_component(g_dlc_path);
                break;
            case ARG_INPUT_LIST:
                inputListFileName = optarg;
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
        switch (g_output_file_type) {
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
            case REWOO_OUTPUT_RAW_RGB: {
                sprintf(path, "%s/frame_%d.raw", g_output_dir.c_str(), decoder->getOuputFrameNum());
                cv::Mat matSrc = cv::Mat(g_video_height * 1.5, g_video_width, CV_8UC1, buf);
                cv::Mat matDst = cv::Mat(g_video_height, g_video_width, CV_8UC3);
                cv::cvtColor(matSrc, matDst, cv::COLOR_YUV2RGB_NV21);
                int32_t cx1 = (g_goalnet_points[2].x + g_goalnet_points[3].x) / 2;
                int32_t cy1 = (g_goalnet_points[2].y + g_goalnet_points[3].y) / 2;
                int32_t lx = std::min(g_video_width - g_output_width,
                                      std::max(0, cx1 - g_output_width / 2));
                int32_t ly = std::min(g_video_height - g_output_height,
                                      std::max(0, cy1 - g_output_height / 2));
                cv::Rect roi(lx, ly, g_output_width, g_output_height);
                cv::Mat crop = matDst(roi);
                cv::Mat floatCrop;
                crop.convertTo(floatCrop, CV_32F, 1 / 255.0);
                FILE *fp = fopen(path, "w+");
                fwrite_ex(floatCrop.data, sizeof(float32_t), floatCrop.total(), 0,
                          floatCrop.elemSize(), fp);
                fwrite_ex(floatCrop.data, sizeof(float32_t), floatCrop.total(), sizeof(float32_t),
                          floatCrop.elemSize(), fp);
                fwrite_ex(floatCrop.data, sizeof(float32_t), floatCrop.total(),
                          sizeof(float32_t) * 2, floatCrop.elemSize(), fp);
                fflush(fp);
                fclose(fp);
                ALOGV("bytes(%lu) written into file %s", floatCrop.total() * floatCrop.elemSize(),
                      path);
                break;
            }
            default:
                break;
        }
    }
}

static std::unique_ptr<SNPE::SNPE> loadModel() {
    // print available runtime
    DlSystem::Runtime_t runtime = checkRuntime();
    switch (runtime) {
        case DlSystem::Runtime_t::GPU:
            ALOGD("Available runtime: GPU");
            break;
        case DlSystem::Runtime_t::CPU:
            ALOGD("Available runtime: CPU");
            break;
        case DlSystem::Runtime_t::DSP:
            ALOGD("Available runtime: DSP");
            break;
        default:
            ALOGD("Available runtime: Unsupported, can not proceed");
            return nullptr;
    }

    // load container
    std::unique_ptr<DlContainer::IDlContainer> container = loadContainerFromFile(g_dlc_path);
    if (container == nullptr) {
        ALOGD("failed to load container, can not proceed");
        return nullptr;
    } else {
        ALOGD("container loaded: %p", container.get());
    }

    // set builder
    DlSystem::RuntimeList runtimeList;
    runtimeList.add(runtime);
    DlSystem::PlatformConfig platformConfig;
    bool usingInitCaching = true;
    return setBuilderOptions(container, runtime, runtimeList,
                             false, platformConfig,
                             usingInitCaching);
}

int main(int argc, char *argv[]) {
    // set dsp library path so that runtime can use dsp
    setenv(DSP_ENV_VAR, DEFAULT_DSP_LIB_DIR, true);

    // handle arguments
    process_opt(argc, argv);

    // 如果指定了模型路径, 则进入模型运行逻辑
    if (!g_dlc_path.empty()) {
        // 如果指定了视频路径,则使用视频解码作为输入,否则使用input list作为输入列表
        if (g_video_path.empty()) {
            g_input_list_path = g_dlc_dir + "/" + inputListFileName;
            run_dlc(new RawListProvider(g_input_list_path));
        } else {
            // load dlc
            std::unique_ptr<SNPE::SNPE> snpe = loadModel();

            // Check the batch size for the container
            // SNPE 1.16.0 (and newer) assumes the first dimension of the tensor shape
            // is the batch size.
            size_t batchSize = 1;
            dumpModel(snpe, &batchSize);

            // run producer
            TensorProducer p(batchSize);
            p.run();

            // run consumer

            // wait done
            while (!g_decode_done || !g_dlc_done) {
                sleep(1);
            }
        }
    }

    // ok
    return EXIT_SUCCESS;
}