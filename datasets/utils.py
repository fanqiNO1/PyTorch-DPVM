import cv2
import numpy as np


def get_image(image_path, is_gray=False):
    image = cv2.imread(image_path)
    if is_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def random_flip(image, mode):
    """
    mode: 1 - No flip
    mode: 2 - Horizontal flip
    mode: 3 - Vertical flip
    mode: 4 - Horizontal and Vertical flip
    """
    if mode == 1:
        return image
    elif mode == 2:
        return cv2.flip(image, 1)
    elif mode == 3:
        return cv2.flip(image, 0)
    elif mode == 4:
        return cv2.flip(image, -1)
    else:
        raise ValueError('Invalid mode for random flip')


def random_rotate(image, mode):
    """
    mode: 1 - No rotation
    mode: 2 - 90 degree rotation
    mode: 3 - 180 degree rotation
    mode: 4 - 270 degree rotation
    """
    if mode == 1:
        return image
    elif mode == 2:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif mode == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif mode == 4:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError('Invalid mode for random rotation')


def to_tensor(image):
    if image.ndim == 2:
        image = image.reshape(image.shape[0], image.shape[1], 1)
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32)
    image = image
    return image
