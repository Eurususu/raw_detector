# 视频提取工具

## 编译方法
使用cmake构建工程，使用make进行编译
由于用到了c++17 的filesystem，需要完全支持c++17的编译器，gcc-7.5可能有问题，经测试，gcc-9.5以上能顺利编译，对应ubuntu distributin 至少是20.04.

```
sudo apt install cmake build-essential libopencv-dev
```
然后安装显卡驱动（尽可能新）和 CUDA 11.x

下载 ropo 源码，cd 进入 repo 目录
```
mkdir build && cd build
cmake ..
make
```

## 使用方法
将私有的.h264和.aqms文件保存为MP4或图片的工具
使用方法：
1. 预览视频
   ```
   ./test_decode -i xxx.h264 -v
   ``` 
   -v 可选，在不指定输出方式或者输出方式无效时都会自动开启-v来显示视频
2. 保存为MP4
    ```
    ./test_decode -i xxx.aqms -o xxx.mp4
    ```
    不检查xxx.mp4是否已经存在，会覆盖原来的文件
    只支持MP4封装，其他视频封装无效
3. 保存为图片
    ```
    ./test_decode -i xxx.aqms -o xxx/xxx
    ```
    保存路径必须已经存在，不然会无法保存
4. 指定保存帧
   ```
    ./test_decode -i xxx.aqms -r 12:1000
    ./test_decode -i xxx.aqms -r 12
    ./test_decode -i xxx.aqms -r :1000
   ```
   默认从第一帧到最后一帧

所有选项可以组合：
```
./test_decode -i xxx.h264 -o xx.mp4 1000:   // 1000帧开始的保存到xx.mp4
./test_decode -i xxx.aqms -o video :1000    // 开头1000帧图片保存到video文件夹
```