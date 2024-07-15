#include "utils.h"
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>

void mkdirs(const char* buf) {
    char tmp[256];
    char *p = NULL;
    size_t len;
    snprintf(tmp, sizeof(tmp), "%s", buf);
    len = strlen(tmp);
    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;
    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, S_IRWXU);
            *p = '/';
        }
    }
    mkdir(tmp, S_IRWXU);
}

bool is_directory(std::string path) {
    struct stat st;
    if(stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        return true;
    }
    return false;
}

bool is_file_exists(std::string path) {
    struct stat st;
    if(stat(path.c_str(), &st) == 0) {
        return true;
    }
    return false;
}