# 1. Control the motor 
## 1.1 Here brltty is sth that prevent to use the ports
To disable the brltty:

```
sudo systemctl stop brltty-udev.service
sudo systemctl mask brltty-udev.service
sudo systemctl stop brltty.service
sudo systemctl disable brltty.service
```
solution source:
[https://forum.arduino.cc/t/solved-tools-serial-port-greyed-out-in-ubuntu-22-04-lts/991568/16](https://forum.arduino.cc/t/solved-tools-serial-port-greyed-out-in-ubuntu-22-04-lts/991568/16)

[https://www.reddit.com/r/pop_os/comments/uf54bi/how_to_remove_or_disable_brltty/](https://www.reddit.com/r/pop_os/comments/uf54bi/how_to_remove_or_disable_brltty/) 

## 1.2 add access to port:
```
sudo su
cd /
cd dev
chown smartlab ttyUSB0
```

# 2. control the camera
## 2.1 install Intel RealSense SDK 2.0

https://github-com.translate.goog/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md?_x_tr_sl=auto&_x_tr_tl=zh-CN&_x_tr_hl=zh-CN

## 2.2 install Python Wrapper
https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
