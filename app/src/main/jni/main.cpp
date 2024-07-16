#include <getopt.h>
#include <stdio.h>
#include "DlEnums.hpp"
#include "DlVersion.hpp"
#include "SNPE.hpp"
#include "SNPEFactory.hpp"
#include "log.h"
#include "codec_api.h"
#include "utils.h"

static const char short_options[] = "hl:v";
static const struct option long_options[] = {
        {"version", no_argument, NULL, 'v'},
        {"dsp_library_path", required_argument, NULL, 'l'},
        {"help", no_argument, NULL, 'h'},
        {0, 0, 0, 0}
};

extern int run_dlc();

DlSystem::Runtime_t checkRuntime() {
    DlSystem::Version_t Version = SNPE::SNPEFactory::getLibraryVersion();
    DlSystem::Runtime_t Runtime;
    ALOGD("Qualcomm (R) Neural Processing SDK Version: %s\n",
          Version.asString().c_str()); //Print Version number
    if (SNPE::SNPEFactory::isRuntimeAvailable(DlSystem::Runtime_t::DSP)) {
        Runtime = DlSystem::Runtime_t::DSP;
    } else if (SNPE::SNPEFactory::isRuntimeAvailable(DlSystem::Runtime_t::GPU)) {
        Runtime = DlSystem::Runtime_t::GPU;
    } else if (SNPE::SNPEFactory::isRuntimeAvailable(DlSystem::Runtime_t::GPU_FLOAT16)) {
        Runtime = DlSystem::Runtime_t::GPU;
    } else if (SNPE::SNPEFactory::isRuntimeAvailable(DlSystem::Runtime_t::CPU)) {
        Runtime = DlSystem::Runtime_t::CPU;
    } else {
        Runtime = DlSystem::Runtime_t::UNSET;
    }
    return Runtime;
}

std::string getRuntimeStr() {
    DlSystem::Runtime_t rt = checkRuntime();
    switch (rt) {
        case DlSystem::Runtime_t::GPU:
            return "GPU";
        case DlSystem::Runtime_t::CPU:
            return "CPU";
        case DlSystem::Runtime_t::DSP:
            return "DSP";
        default:
            return "Unsupported";
    }
}

static void print_version() {
    DlSystem::Version_t libVer = SNPE::SNPEFactory::getLibraryVersion();
    printf("hexagon version: %s, available runtime: %s\n", libVer.toString().c_str(),
           getRuntimeStr().c_str());
}

static void print_usage(int argc, char *argv[]) {
    fprintf(stdout,
            "Usage: %s [options]\n"
            "Version %s\n"
            "Options:\n"
            "-h | --help\t\t\tprint help\n"
            "-l | --dsp_library_path\t\tset ADSP_LIBRARY_PATH environment, default is /vendor/lib\n"
            "-v | --version\t\t\tprint version \n"
            "\n",
            argv[0], "v1.0.0");
}

static void process_opt(int argc, char *argv[]) {
    for (;;) {
        int idx;
        int c;
        c = getopt_long(argc, argv, short_options, long_options, &idx);
        if (-1 == c)
            break;
        switch (c) {
            case 'h':
                print_usage(argc, argv);
                exit(EXIT_SUCCESS);
            case 'v':
                print_version();
                exit(EXIT_SUCCESS);
            case 'l':
                setenv("ADSP_LIBRARY_PATH", optarg, true);
                break;
            default:
                break;
        }
    }
}

const std::string outDir = "/data/local/tmp/decode_output";
static void onOutputAvailable(
        AMediaCodec *codec,
        Decoder *decoder,
        int32_t index,
        AMediaCodecBufferInfo *bufferInfo) {
    size_t bufSize;
    uint8_t *buf = AMediaCodec_getOutputBuffer(codec, index, &bufSize);
    if (buf && bufferInfo->size > 0) {
        char path[512] = {0};
        sprintf(path, "%s/frame_%d.yuv", outDir.c_str(), decoder->getOuputFrameNum());
        FILE* fp = fopen(path, "w+");
        fwrite(buf, sizeof(char), bufferInfo->size, fp);
        fflush(fp);
        fclose(fp);
        ALOGV("bytes(%d) written into file %s", bufferInfo->size, path);
    }
}

int main(int argc, char *argv[]) {
    // set dsp library path so that runtime can use dsp
    setenv("ADSP_LIBRARY_PATH", "/vendor/lib", true);

    // handle arguments
    process_opt(argc, argv);

    if(!outDir.empty() && !is_directory(outDir)) {
        mkdirs(outDir.c_str());
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