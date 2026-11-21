"""Only write the upstream mouth-mask footprint; preserve other pixels exactly."""
import cv2
import numpy as np


def composite(original, normalized_face, matrix, box, fixed_mask):
    x1, y1, x2, y2 = box
    width, height = int(x2-x1), int(y2-y1)
    face = np.clip((normalized_face + 1) * 127.5, 0, 255)
    face = cv2.resize(face, (width, height), interpolation=cv2.INTER_CUBIC)
    alpha = cv2.resize(1-fixed_mask, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0, borderType=cv2.BORDER_CONSTANT)
    inverse = cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float32).reshape(2, 3))
    size = (original.shape[1], original.shape[0])
    face = cv2.warpAffine(face, inverse, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    alpha = cv2.warpAffine(alpha, inverse, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    alpha = np.clip(alpha, 0, 1)[..., None]
    return np.round(face * alpha + original * (1-alpha)).clip(0, 255).astype(np.uint8)
