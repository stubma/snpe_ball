package com.example.hexagon_test

import android.app.Application
import android.system.Os

class MyApp : Application() {
    override fun onCreate() {
        super.onCreate()

//        Os.setenv("LD_LIBRARY_PATH", "/data/local/tmp/snpe/lib", true)
//        Os.setenv("ADSP_LIBRARY_PATH", "/data/local/tmp/snpe/lib", true)
    }
}