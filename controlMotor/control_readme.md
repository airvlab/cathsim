# 1.Here brltty is sth that prevent to use the ports
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

# 2. add access to port:
```
sudo su
cd /
cd dev
chown smartlab ttyUSB0
```

