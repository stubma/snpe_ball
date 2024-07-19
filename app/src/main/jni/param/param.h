// Copyright 2021 Rockchip Electronics Co., Ltd. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include "iniparser.h"

#if __cplusplus
extern "C" {
#endif

extern dictionary *g_ini_d_;

int rk_param_exist(const char *entry);
int rk_param_get_int(const char *entry, int default_val);
int rk_param_set_int(const char *entry, int val);
unsigned int rk_param_get_color(const char* entry, unsigned int default_val);
void rk_param_get_color_array(const char* entry, unsigned int* arr, int capacity, int* num);
const char *rk_param_get_string(const char *entry, const char *default_val);
int rk_param_set_string(const char *entry, const char *val);
void rk_param_get_int_array(const char* entry, int* arr, int capacity, int* num);
void rk_param_get_float_array(const char* entry, float* arr, int capacity, int* num);
void rk_param_get_double_array(const char* entry, double* arr, int capacity, int* num);
float rk_param_get_float(const char *entry, float default_val);
int rk_param_save();
int rk_param_init(char *ini_path);
int rk_param_deinit();
int rk_param_reload();

void rk_param_inject_clear();
void rk_param_inject_remove(const char* entry);
void rk_param_inject_int(const char* entry, int val);
void rk_param_inject_string(const char *entry, const char *val);

#if __cplusplus
}
#endif