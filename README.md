# snpe_ball
snpe track ball

* testdata/ball_v2 push到设备的/data/local/tmp
* 在android studio里面rebuild project编译
* 生成的可执行文件会在app./build/intermediates/cxx/Debug/2y4432u2/obj/arm64-v8a/hexagon这样的路径下, 可以在app下面用find搜一下
* 将hexagon push到/data/local/tmp
* adb remount /vendor, 使vendor切换到overlay文件系统, 然后adb reboot
* 重启完成后/vendor就应该切换到overlay fs了, 可以用df看一下. 然后使用adb shell 'mount -o rw,remount /vendor'让vendor进入可写状态
* testdata/snpe/lib下面的库(除了libSnpeHtpV73Skel.so) push到/vendor/lib64
* testdata/snpe/lib/libSnpeHtpV73Skel.so push到/vendor/lib
* opencv/native/libs/arm64-v8a下面的库push到/vendor/lib64
* 编辑/vendor/etc/public.libraries.txt文件, 在最后添加libSnpeHtpV73Skel.so和libopencv_java4.so
* 重启板子, 直接执行hexagon即可, 应该就能正常运行了

# ball_v3的运行方法

* 将testdata/ball_v3 push到设备任意目录, 假设是/data/local/tmp/ball_v3
* 在android studio里面rebuild project编译
* 生成的可执行文件会在app./build/intermediates/cxx/Debug/2y4432u2/obj/arm64-v8a/hexagon这样的路径下, 可以在app下面用find搜一下
* 将hexagon push到/data/local/tmp/ball_v3
* 执行export LD_LIBRARY_PATH=/data/local/tmp/ball_v3/lib
* 将要解码的视频push到/data/local/tmp/ball_v3
* 执行 ./hexagon即可, 其需要的参数已经配置在了cfg.ini里, cfg.ini和hexagon必须在同一目录, 如果有参数要调整请修改cfg.ini, 请看cfg.ini的注释