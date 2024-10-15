
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