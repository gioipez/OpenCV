import cv2

def save_result(out_img, output_filename) -> bool:
    return cv2.imwrite(output_filename, out_img)