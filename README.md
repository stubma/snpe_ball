# snpe_ball
snpe track ball

1. testdata/snpe这个目录push到设备的/data/local/tmp
2. testdata/ball_v2 push到设备的/data/local/tmp
3. 在android studio里面rebuild project编译
4. 生成的可执行文件会在app./build/intermediates/cxx/Debug/2y4432u2/obj/arm64-v8a/hexagon这样的路径下, 可以在app下面用find搜一下
5. 将hexagon push到/data/local/tmp
6. adb shell到设备里, 设置一下环境变量
export ADSP_LIBRARY_PATH=/data/local/tmp/snpe/lib
export LD_LIBRARY_PATH=/data/local/tmp/snpe/lib

7. 执行hexagon即可
