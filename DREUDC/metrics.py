import numpy as np
import math
from skimage.metrics import structural_similarity as ssim


def PSNR(original, compressed):
    original = original.astype(np.float32)
    compressed = compressed.astype(np.float32)

    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 1.
    psnr = 20 * np.log10(max_pixel / math.sqrt(mse))
    return psnr


def SSIM(img1, img2):
    ssim_val = ssim(img1, img2, gaussian_weights=True, use_sample_covariance=False, multichannel=True)
    return ssim_val
