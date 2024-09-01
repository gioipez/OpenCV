import rawpy
import cv2
from utils.opencvLogger import logger


class rawImage:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_raw_image_bgr(self):
        with rawpy.imread(self.file_path) as raw:
            rgb = raw.postprocess(
                no_auto_bright=True,
                output_bps=16,
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB
                )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    logger.info("Reading raw image")
    raw_image = rawImage('/Users/giovannilopez/Downloads/2024-08-15_Cultivos/camara1/phenotype_1_15062153_andigena/flower_DSC09050.ARW')
    raw_image = raw_image.read_raw_image_bgr()

    # Mostrar la imagen usando OpenCV
    cv2.imshow('RAW Image', raw_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
