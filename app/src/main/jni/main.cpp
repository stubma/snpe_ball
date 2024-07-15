#include <getopt.h>
#include <stdio.h>
#include "DlEnums.hpp"
#include "DlVersion.hpp"
#include "SNPE.hpp"
#include "SNPEFactory.hpp"
#include "log.h"
#include "codec_api.h"

static const char short_options[] = "v";
static const struct option long_options[] = {
        {"version", no_argument, NULL, 'v'},
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

static void process_opt(int argc, char *argv[]) {
    for (;;) {
        int idx;
        int c;
        c = getopt_long(argc, argv, short_options, long_options, &idx);
        if (-1 == c)
            break;
        switch (c) {
            case 'v':
                print_version();
                exit(EXIT_SUCCESS);
            default:
                break;
        }
    }
}

int main(int argc, char *argv[]) {
    // handle arguments
    process_opt(argc, argv);

    decode_video(
            "/data/local/tmp/MediaBenchmark/res/",
            "rewoo_full.mp4",
            "/data/local/tmp/decoder.stat",
            "c2.qti.avc.decoder",
            false,
            "/data/local/tmp/decode_output");

    // ok
    return EXIT_SUCCESS;
}