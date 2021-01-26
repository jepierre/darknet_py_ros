# Darknet ROS Python Node

# todo: 
addinstructions how to build darknet
publish x,  y, z coordinates of detections


```
cd ~/catkin_ws/src
git clone --recursive https://github.com/jepierre/darknet_py_ros.git

git checkout dev
bash build_darknet.sh

roslaunch darknet_py_ros  darknet_py_ros_.launch
```

To maximize performance, make sure to build in Release mode. You can specify the build type by setting

catkin_make -DCMAKE_BUILD_TYPE=Release -j8