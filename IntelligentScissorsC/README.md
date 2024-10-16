


# Build

```shell
g++ -std=c++11 -o intelligent_scissors intelligent_scissors.cpp \
-I/usr/local/include/opencv4 \
-L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
```

# Usage

`Usage: ./intelligent_scissors <image_path> <output_directory>`

```
./intelligent_scissors /Users/giovannilopez/Downloads/2024-08-15_Cultivos/calibrated/flower_DSC_4436_JPG.jpg /Users/giovannilopez/Downloads/2024-08-15_Cultivos/segmented_images_intelligent_scissors
```