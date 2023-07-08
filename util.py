import cv2
import numpy as np


class BackgroundImages:
    BG_NUM = 5
    def __init__(self) -> None:
        self.bgs = []
        for i in range(BackgroundImages.BG_NUM):
            self.bgs.append(cv2.imread(f'static/bg_sources/bg{i}.jpg',cv2.IMREAD_ANYCOLOR))
    
    def get_bg(self, num):
        return self.bgs[num].copy()
        


def load_sample_img()->np.ndarray:
    img = cv2.imread('sample_data/lena.jpg')
    return img

def preprocessing(img:np.ndarray, h:int, w:int)->np.ndarray:
    img = np.array(img).astype('float32')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(src=img, dsize=(h, w), interpolation=cv2.INTER_AREA)
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    return img

def convert_result_to_image(stylized_image, h:int, w:int) -> np.ndarray:
    stylized_image = stylized_image.squeeze().transpose(1, 2, 0)
    stylized_image = cv2.resize(src=stylized_image, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    stylized_image = np.clip(stylized_image, 0, 255).astype(np.uint8)
    stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_BGR2RGB)
    return stylized_image


def convert_upscale_result_to_image(result) -> np.ndarray:
    """
    Convert network result of floating point numbers to image with integer
    values from 0-255. Values outside this range are clipped to 0 and 255.

    :param result: a single superresolution network result in N,C,H,W shape
    """
    result = result.squeeze(0).transpose(1, 2, 0)
    result *= 255
    result[result < 0] = 0
    result[result > 255] = 255
    result = result.astype(np.uint8)
    return result

def preprocessing_bg_removal(image):
    resized_image = cv2.resize(src=image, dsize=(512, 512))
    # Convert the image shape to a shape and a data type expected by the network
    # for OpenVINO IR model: (1, 3, 512, 512).
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
    
    return input_image

def postprocessing_bg_removal(bg_remove_result, original_image, bg):
    resized_result = np.rint(
        cv2.resize(src=np.squeeze(bg_remove_result), dsize=(original_image.shape[1], original_image.shape[0]))
    ).astype(np.uint8)
    bg_removed_result = original_image.copy()
    bg_removed_result[resized_result == 0] = 255
    background_image = cv2.cvtColor(src=bg, code=cv2.COLOR_BGR2RGB)
    background_image = cv2.resize(src=background_image, dsize=(original_image.shape[1], original_image.shape[0]))

    background_image[resized_result == 1] = 0
    new_image = background_image + bg_removed_result
    return new_image