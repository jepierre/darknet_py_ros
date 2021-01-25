#!/bin/bash

# set up Makefile
sed -i 's/GPU=0/GPU=1/g' darknet/Makefile
sed -i 's/CUDNN=0/CUDNN=1/g' darknet/Makefile
sed -i 's/ZED_CAMERA=0/ZED_CAMERA=1/g' darknet/Makefile
sed -i 's/LIBSO=0/LIBSO=1/g' darknet/Makefile
## don't use opencv in Darknet or you'll run into issues with python3 cv2
# sed -i 's/OPENCV=0/OPENCV=1/g' darknet_py_ros/darknet/Makefile

# Build Darknet
cd darknet
make -j8
