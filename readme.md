# 视频提取工具

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