
# Determine prominent color of a Flower with an Original Image and a segmented one

Using the FLASK Service, it will ask you the name of the flower and the segmented image, like this:

<img width="554" alt="image" src="https://github.com/user-attachments/assets/82e72fc9-55b5-4eee-9e30-888915375bdb">

> NOTE: Images should be stored under `image` directory. In my case, sitting in the root folder, it will be something like `Services/image/`

## This is the beuty result: 

<img width="1084" alt="image" src="https://github.com/user-attachments/assets/ff50caf6-f6f5-4ca3-aabe-deb281e9cdfe">

You can run with this:

```shell
python -m Services.app
```

# OpenCV

This repo contain code to run OpenCV [core module](https://pypi.org/project/opencv-python/) and [contrib](https://pypi.org/project/opencv-contrib-python/) in Python

# Install

## Prerequisite

Install CMAKE

```
brew install cmake 
```

1. Create a temporary directory, which we denote as `build_opencv`, where you want to put the generated Makefiles, project files as well the object files and output binaries and enter there.

```
mkdir build_opencv
cd build_opencv
```

2. Configuring

```
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DOPENCV_EXTRA_MODULES_PATH="/Users/giovannilopez/Universidad/opencv_contrib/modules" -DOpenCV_DIR="/Users/giovannilopez/opencv" ../opencv
```

3. Install

```
make install -j
```

# Troubleshoot

## DYLib

Error

```
dyld[46328]: Library not loaded: @rpath/libopencv_core.410.dylib
```

Solution

```
export DYLD_LIBRARY_PATH=/usr/local/lib/:$DYLD_LIBRARY_PATH
```

# Reference

* [MacOS Install](https://docs.opencv.org/4.x/d0/db2/tutorial_macos_install.html)
