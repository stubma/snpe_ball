// Copyright 2021 Rockchip Electronics Co., Ltd. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include "common.h"
#include "iniparser.h"
#include "rklog.h"
#include <sys/stat.h>

#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "param.c"

#if __cplusplus
extern "C" {
#endif

char g_ini_path_[256];
dictionary *g_ini_d_ = NULL;
static pthread_mutex_t g_param_mutex = PTHREAD_MUTEX_INITIALIZER;

// 用来注入一些假值，这样可以不用修改rkipc.ini文件而实现配置变化, 因为有时候某些配置值需要
// 重启或者重新初始化才能生效，通过注入假值，可以不用重启或者初始化
dictionary* g_ini_inject_ = NULL;

int rk_param_dump() {
	const char *section_name;
	const char *keys[1024];
	int section_keys;
	int section_num = iniparser_getnsec(g_ini_d_);
	LOG_DEBUG("section_num is %d\n", section_num);

	LOG_INFO("==================section_num========param.c====line24=======%d===\n",section_num);

	for (int i = 0; i < section_num; i++) {
		section_name = iniparser_getsecname(g_ini_d_, i);
		LOG_DEBUG("section_name is %s\n", section_name);
		section_keys = iniparser_getsecnkeys(g_ini_d_, section_name);
		for (int j = 0; j < section_keys; j++) {
			iniparser_getseckeys(g_ini_d_, section_name, keys);
			LOG_DEBUG("%s = %s\n", keys[j], iniparser_getstring(g_ini_d_, keys[j], ""));
		}
	}

	return 0;
}

int rk_param_save() {
	FILE *fp = fopen(g_ini_path_, "w");
	if (fp == NULL) {
		LOG_ERROR("%s, fopen error!\n", g_ini_path_);
		iniparser_freedict(g_ini_d_);
		g_ini_d_ = NULL;
		return -1;
	}
	iniparser_dump_ini(g_ini_d_, fp);

	fflush(fp);
	fclose(fp);

	return 0;
}

void rk_param_inject_clear() {
	if(g_ini_inject_) {
		iniparser_freedict(g_ini_inject_);
		g_ini_inject_ = NULL;
	}
	g_ini_inject_ = dictionary_new(0);
}

void rk_param_inject_remove(const char* entry) {
	pthread_mutex_lock(&g_param_mutex);
	iniparser_unset(g_ini_inject_, entry);
	pthread_mutex_unlock(&g_param_mutex);
}

void rk_param_inject_int(const char* entry, int val) {
	char tmp[8];
	sprintf(tmp, "%d", val);
	pthread_mutex_lock(&g_param_mutex);
	iniparser_set(g_ini_inject_, entry, tmp);
	pthread_mutex_unlock(&g_param_mutex);
}

void rk_param_inject_string(const char *entry, const char *val) {
	pthread_mutex_lock(&g_param_mutex);
	iniparser_set(g_ini_inject_, entry, val);
	rk_param_save();
	pthread_mutex_unlock(&g_param_mutex);
}

int rk_param_get_int(const char *entry, int default_val) {
	int ret;
	pthread_mutex_lock(&g_param_mutex);
	ret = iniparser_getint(g_ini_d_, entry, default_val);
	pthread_mutex_unlock(&g_param_mutex);

	return ret;
}

int rk_param_set_int(const char *entry, int val) {
	char tmp[8];
	sprintf(tmp, "%d", val);
	pthread_mutex_lock(&g_param_mutex);
	iniparser_set(g_ini_d_, entry, tmp);
	rk_param_save();
	pthread_mutex_unlock(&g_param_mutex);

	return 0;
}

void rk_param_get_float_array(const char* entry, float* arr, int capacity, int* num) {
	const char *str;
	pthread_mutex_lock(&g_param_mutex);
	str = iniparser_getstring(g_ini_d_, entry, "");
	pthread_mutex_unlock(&g_param_mutex);
	size_t len = strlen(str);
	if(len <= 0) return;

	// strtok will overwrite source so we make a copy
	char* buf = (char*)malloc(len + 1);
	strcpy(buf, str);

	// parse by delimiter
	*num = 0;
	char* token = strtok(buf, ",");
	while(token && *num < capacity) {
		float f;
		sscanf(token, "%f", &f);
		arr[*num] = f;
		(*num)++;
		token = strtok(NULL, ",");
	}
	
	// free
	free(buf);
}

void rk_param_get_double_array(const char* entry, double* arr, int capacity, int* num) {
	const char *str;
	pthread_mutex_lock(&g_param_mutex);
	str = iniparser_getstring(g_ini_d_, entry, "");
	pthread_mutex_unlock(&g_param_mutex);
	size_t len = strlen(str);
	if(len <= 0) return;

	// strtok will overwrite source so we make a copy
	char* buf = (char*)malloc(len + 1);
	strcpy(buf, str);

	// parse by delimiter
	*num = 0;
	char* token = strtok(buf, ",");
	while(token && *num < capacity) {
		double f;
		sscanf(token, "%lf", &f);
		arr[*num] = f;
		(*num)++;
		token = strtok(NULL, ",");
	}
	
	// free
	free(buf);
}

void rk_param_get_int_array(const char* entry, int* arr, int capacity, int* num) {
	const char *str;
	pthread_mutex_lock(&g_param_mutex);
	str = iniparser_getstring(g_ini_d_, entry, "");
	pthread_mutex_unlock(&g_param_mutex);
	size_t len = strlen(str);
	if(len <= 0) return;

	// strtok will overwrite source so we make a copy
	char* buf = (char*)malloc(len + 1);
	strcpy(buf, str);

	// parse by delimiter
	*num = 0;
	char* token = strtok(buf, ",");
	while(token && *num < capacity) {
		arr[*num] = atoi(token);
		(*num)++;
		token = strtok(NULL, ",");
	}
	
	// free
	free(buf);
}

float rk_param_get_float(const char *entry, float default_val) {
	// get string
	const char *str;
	pthread_mutex_lock(&g_param_mutex);
	str = iniparser_getstring(g_ini_d_, entry, "0");
	pthread_mutex_unlock(&g_param_mutex);

	// parse float
	float ret;
	sscanf(str, "%f", &ret);
	return ret;
}

int rk_param_exist(const char *entry) {
	const char *ret;
	pthread_mutex_lock(&g_param_mutex);
	ret = iniparser_getstring(g_ini_d_, entry, "");
	pthread_mutex_unlock(&g_param_mutex);
	return ret && strlen(ret) > 0;
}

unsigned int rk_param_get_color(const char* entry, unsigned int default_val) {
	const char *str;
	pthread_mutex_lock(&g_param_mutex);
	str = iniparser_getstring(g_ini_d_, entry, "");
	pthread_mutex_unlock(&g_param_mutex);
	int len = strlen(str);

	unsigned int color = 0;
	if(len <= 0) {
		color = default_val;
	} else {
		sscanf(str, "%x", &color);
		if(len <= 6) {
			color |= 0xff000000u;
		}
	}
	return color;
}

void rk_param_get_color_array(const char* entry, unsigned int* arr, int capacity, int* num) {
	const char *str;
	pthread_mutex_lock(&g_param_mutex);
	str = iniparser_getstring(g_ini_d_, entry, "");
	pthread_mutex_unlock(&g_param_mutex);
	size_t len = strlen(str);
	if(len <= 0) return;

	// strtok will overwrite source so we make a copy
	char* buf = (char*)malloc(len + 1);
	strcpy(buf, str);

	// parse by delimiter
	*num = 0;
	unsigned int color = 0;
	char* token = strtok(buf, ",");
	while(token && *num < capacity) {
		len = strlen(token);
		if(len > 0) {
			sscanf(token, "%x", &color);
			if(len <= 6) {
				color |= 0xff000000u;
			}
		}
		arr[*num] = color;
		(*num)++;
		token = strtok(NULL, ",");
	}
	
	// free
	free(buf);
}

const char *rk_param_get_string(const char *entry, const char *default_val) {
	const char *ret;
	pthread_mutex_lock(&g_param_mutex);
	ret = iniparser_getstring(g_ini_d_, entry, default_val);
	pthread_mutex_unlock(&g_param_mutex);

	return ret;
}

int rk_param_set_string(const char *entry, const char *val) {
	pthread_mutex_lock(&g_param_mutex);
	iniparser_set(g_ini_d_, entry, val);
	rk_param_save();
	pthread_mutex_unlock(&g_param_mutex);

	return 0;
}

static int is_file_exists(const char* path) {
	struct stat st;
	if(stat(path, &st) == 0) {
		return 1;
	}
	return 0;
}

int rk_param_init(char *ini_path) {
	LOG_DEBUG("%s\n", __func__);
	pthread_mutex_lock(&g_param_mutex);

	// init inject dict
	if(!g_ini_inject_) {
		g_ini_inject_ = dictionary_new(0);
	}

	// rkipc.ini should be in /oem/usr/share
	if (ini_path) {
		memcpy(g_ini_path_, ini_path, strlen(ini_path));
	} else {
		const char* ini1 = "/oem/usr/share/rkipc.ini";
		memcpy(g_ini_path_, ini1, strlen(ini1));
	}
	LOG_INFO("g_ini_path_ is %s\n", g_ini_path_);

	// load ini
	dictionary* new_dict = iniparser_load(g_ini_path_);
	if (new_dict == NULL) {
		LOG_ERROR("iniparser_load error!\n");
		pthread_mutex_unlock(&g_param_mutex);
		return -1;
	}

	// if success, reload old and save new dict
	if(g_ini_d_) {
		iniparser_freedict(g_ini_d_);
		g_ini_d_ = NULL;
	}
	g_ini_d_ = new_dict;

	rk_param_dump();
	pthread_mutex_unlock(&g_param_mutex);

	return 0;
}

int rk_param_deinit() {
	LOG_INFO("%s\n", __func__);
	if (g_ini_d_ == NULL)
		return 0;
	pthread_mutex_lock(&g_param_mutex);
	rk_param_save();
	if (g_ini_d_) {
		iniparser_freedict(g_ini_d_);
		g_ini_d_ = NULL;
	}
	if (g_ini_inject_) {
		iniparser_freedict(g_ini_inject_);
		g_ini_inject_ = NULL;
	}
	pthread_mutex_unlock(&g_param_mutex);

	return 0;
}

int rk_param_reload() {
	LOG_INFO("%s\n", __func__);
	pthread_mutex_lock(&g_param_mutex);
	if (g_ini_d_)
		iniparser_freedict(g_ini_d_);
	g_ini_d_ = iniparser_load(g_ini_path_);
	if (g_ini_d_ == NULL) {
		LOG_ERROR("iniparser_load error!\n");
		pthread_mutex_unlock(&g_param_mutex);
		return -1;
	}
	rk_param_dump();
	pthread_mutex_unlock(&g_param_mutex);

	return 0;
}

#if __cplusplus
}
#endif