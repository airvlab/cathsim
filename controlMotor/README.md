# Burt the board

## Download the ardurino

 downlod from website
 set it as excutive file
 double click it

## Release the port

### Method 1 : manually

sudo systemctl stop brltty-udev.service
sudo systemctl mask brltty-udev.service
sudo systemctl stop brltty.service
sudo systemctl disable brltty.service

solution source:
[https://forum.arduino.cc/t/solved-tools-serial-port-greyed-out-in-ubuntu-22-04-lts/991568/16](https://forum.arduino.cc/t/solved-tools-serial-port-greyed-out-in-ubuntu-22-04-lts/991568/16)

[https://www.reddit.com/r/pop_os/comments/uf54bi/how_to_remove_or_disable_brltty/](https://www.reddit.com/r/pop_os/comments/uf54bi/how_to_remove_or_disable_brltty/)

### Run the file

bash port_release.sh

## Get the permission of port

### Manually

sudo su
cd /dev/
chown `<username>` ttyUSB0

### Run the File

replace the username in the permission file

bash permission.sh

## Verify and load the actuator.ino file

make sure you are sudoer

## Install the camera's app

https://github-com.translate.goog/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md?_x_tr_sl=auto&_x_tr_tl=zh-CN&_x_tr_hl=zh-CN

sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null

`sudo apt-get install apt-transport-https`

echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \

sudo tee /etc/apt/sources.list.d/librealsense.list

sudo apt-get update

`sudo apt-get install librealsense2-dkms`

`sudo apt-get install librealsense2-utils`

 verify the install

`realsense-viewer`

## Install the python wrap of camera

https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python

pip install pyrealsense2

# Install miniconda

https://docs.anaconda.com/miniconda/#quick-command-line-install

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

# Create the environment

conda create -n train python =3.9

conda activate train

pip install pyserial numpy pyrealsense2 opencv-python
